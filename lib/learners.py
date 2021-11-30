import logging
import typing
from dataclasses import dataclass
from enum import Enum

import dill
import numpy
import torch
from box import Box
from opacus import PrivacyEngine
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib import os_utils
from lib.os_utils import loss_logger_helper

logger = logging.getLogger()


class LearnerModelSavingStrategy(Enum):
    NEVER = "never"
    EVERY_ROUND = "per_round"
    EVERY_EPOCH = "per_epoch"


class LearnerEvaluationStrategy(Enum):
    NEVER = "never"
    EVERY_ROUND = "per_round"
    EVERY_EPOCH = "per_epoch"


def get_optimizer(model, optimizer="adam", lr=1e-3, **kwargs):
    if isinstance(model, nn.Module):
        params = model.parameters()
    else:
        params = model
    if optimizer == "adam":
        weight_decay = kwargs.get("weight_decay", 0)
        optimizer = optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        weight_decay = kwargs.get("weight_decay", 0)
        momentum = kwargs.get("momentum", 0)
        nesterov = False if momentum == 0 else kwargs.get("nesterov", False)
        optimizer = optim.SGD(
            params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov
            )
    else:
        raise Exception(f"{optimizer} not implemented")

    if kwargs.get("private_training", False):
        privacy_engine = PrivacyEngine(
            model, kwargs["virtual_batch_size"], kwargs["sample_size"], alphas=list(range(2, 32)),
            noise_multiplier=kwargs["noise_multiplier"], max_grad_norm=kwargs["grad_norm"]
            )
        privacy_engine.attach(optimizer)

    return optimizer


@dataclass
class LearnerMetaData:
    model: nn.Module
    optimizer_params: Box
    id: int
    train_loader: DataLoader
    valid_loader: typing.Optional[DataLoader] = None
    result_dir: typing.Optional[str] = None
    log_every: int = 100
    model_saving_strategy: LearnerModelSavingStrategy = LearnerModelSavingStrategy.NEVER
    evaluation_strategy: LearnerEvaluationStrategy = LearnerEvaluationStrategy.NEVER,
    scale: float = 0.0,
    unique_gradient_learner: bool = False

    def create_learner(self):
        if self.unique_gradient_learner:
            return ProjectionLearner(
                self.model,
                self.optimizer_params,
                id=self.id,
                train_loader=self.train_loader,
                valid_loader=self.valid_loader,
                result_dir=self.result_dir,
                log_every=self.log_every,
                model_saving_strategy=self.model_saving_strategy,
                evaluation_strategy=self.evaluation_strategy,
                scale=self.scale
                )
        return Learner(
            self.model,
            self.optimizer_params,
            id=self.id,
            train_loader=self.train_loader,
            valid_loader=self.valid_loader,
            result_dir=self.result_dir,
            log_every=self.log_every,
            model_saving_strategy=self.model_saving_strategy,
            evaluation_strategy=self.evaluation_strategy,
            )


class Learner:
    def __init__(
            self, model: nn.Module,
            optimizer_params: typing.Optional[Box],
            id: int,
            train_loader: typing.Optional[DataLoader] = None,
            valid_loader: typing.Optional[DataLoader] = None,
            result_dir: typing.Optional[str] = None,
            log_every=100,
            model_saving_strategy=LearnerModelSavingStrategy.NEVER,
            evaluation_strategy=LearnerEvaluationStrategy.NEVER
            ):
        self.model = model

        self.optimizer_params = optimizer_params
        self.optimizer = None
        self.id = id
        self.result_dir = result_dir
        self.step = 0
        self.log_every = log_every
        self.model_saving_strategy = model_saving_strategy
        self.evaluation_strategy = evaluation_strategy

        # logging stuff
        if result_dir is not None:
            os_utils.safe_makedirs(result_dir)
            self.summary_writer = SummaryWriter(log_dir=result_dir)

        self.train_loader = train_loader
        self.valid_loader = valid_loader

    def train(self, *, num_epochs, epoch_offset, round, device="cpu", **kwargs):
        logger.info(f"Starting to train learner: {self.id}")
        self.on_train_begin(
            num_epochs=num_epochs, epoch_offset=epoch_offset, round=round, device=device, **kwargs
            )

        for epoch in range(epoch_offset, epoch_offset + num_epochs):
            self.on_epoch_begin(epoch=epoch, **kwargs)

            # train loop
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                pred = self.model(batch)
                loss, aux_loss = self.model.loss(pred, batch, reduce=True)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.step += 1

                loss_logger_helper(
                    loss, aux_loss, writer=self.summary_writer,
                    step=self.step, epoch=epoch,
                    log_every=self.log_every, string="train"
                    )

            self.on_epoch_end(epoch=epoch, **kwargs)

        self.on_train_end(
            num_epochs=num_epochs, epoch_offset=epoch_offset, round=round, **kwargs
            )
        logger.info(f"Finished training learner: {self.id}")

    def evaluate(self, loader, **kwargs):
        losses = []
        aux_losses = {}
        self.model.eval()

        with torch.no_grad():
            for i, batch in enumerate(loader):
                pred = self.model(batch)
                loss, aux_loss = self.model.loss(pred, batch, reduce=False)

                losses.extend(loss.cpu().tolist())
                # collect stats
                if i == 0:
                    for k, v in aux_loss.items():
                        # when we can't return sample wise statistics, we need to do this
                        if len(v.shape) == 0:
                            aux_losses[k] = [v.cpu().tolist()]
                        else:
                            aux_losses[k] = v.cpu().tolist()
                else:
                    for k, v in aux_loss.items():
                        if len(v.shape) == 0:
                            aux_losses[k].append(v.cpu().tolist())
                        else:
                            aux_losses[k].extend(v.cpu().tolist())

            return numpy.mean(losses), {k: numpy.mean(v) for (k, v) in aux_losses.items()}

    def on_train_begin(self, *, device, **kwargs):
        logger.debug("In train begin")
        if self.optimizer_params is None:
            raise Exception("Cannot train without optimizer params")

        if self.optimizer is None or self.optimizer_params.reset_optimizer:
            self.optimizer = get_optimizer(self.model, **self.optimizer_params)

        self.model.to(device)

    def on_train_end(self, *, epoch_offset, num_epochs, round, **kwargs):
        logger.debug("In train end")

        if self.evaluation_strategy == LearnerEvaluationStrategy.EVERY_ROUND:
            if self.valid_loader:
                loss, aux_loss = self.evaluate(self.valid_loader)
                loss_logger_helper(
                    loss, aux_loss, writer=self.summary_writer, step=self.step,
                    epoch=epoch_offset + num_epochs, log_every=self.log_every,
                    string="val", force_print=True
                    )
            else:
                logger.info(
                    f"Didnot evaluate learner {self.id} at the end of round because data loader not found "
                    )

        if self.model_saving_strategy == LearnerModelSavingStrategy.EVERY_ROUND:
            self.save(f"{self.result_dir}/model_round_{round}.pt")

        # free the gpu
        self.model.to("cpu")

    def on_epoch_begin(self, **kwargs):
        logger.debug("In epoch begin")
        pass

    def on_epoch_end(self, *, epoch, **kwargs):
        logger.debug("In epoch end")
        if self.evaluation_strategy == LearnerEvaluationStrategy.EVERY_EPOCH:
            if self.valid_loader:
                loss, aux_loss = self.evaluate(self.valid_loader)
                loss_logger_helper(
                    loss, aux_loss, writer=self.summary_writer, step=self.step, epoch=epoch,
                    log_every=self.log_every, string="val", force_print=True
                    )
            else:
                logger.info(
                    f"Didnot evaluate learner {self.id} at the end of epoch because data loader not found "
                    )

        if self.model_saving_strategy == LearnerModelSavingStrategy.EVERY_EPOCH:
            self.save(f"{self.result_dir}/model_epoch_{epoch}.pt")

    def set_params(self, params):
        self.model.load_state_dict(params)

    def get_params(self):
        return self.model.state_dict()

    def save(self, fname, **kwargs):
        kwargs.update({"model": self.model.state_dict(), })
        torch.save(kwargs, open(fname, "wb"), pickle_module=dill)


def project_gradients(grads):
    """
    create basis for every gradient using gram-schmidt orthogonalization and then project
    the gradient on that basis
    """
    old_shape = grads.shape
    n = grads.shape[0]
    grads = grads.reshape((n, -1))
    proj = torch.zeros_like(grads)

    for cur_idx in range(n):
        # construct a basis with other gradients
        basis = torch.zeros_like(grads)
        for new_idx in range(n):
            if new_idx == cur_idx:
                continue
            if new_idx > 0:
                coords = torch.mm(basis[:new_idx], grads[new_idx].view((-1, 1)))  # (new_idx, 1)
                rem = grads[new_idx].view((-1, 1)) - torch.mm(basis[:new_idx].T, coords)  # (d, 1)
                rem = rem.flatten()  # (d,)
            else:
                rem = grads[new_idx]
            # update the basis
            basis[new_idx] = rem / (torch.norm(rem) + 1e-9)

        # project current gradient
        coords = torch.mm(basis, grads[cur_idx].view((-1, 1)))  # (n, 1)
        proj[cur_idx] = torch.mm(basis.T, coords).flatten()

    return proj.reshape(old_shape)


class ProjectionLearner(Learner):
    # This is like skorch but instead of callbacks we use class functions (looks less magic)
    # this is an evolving template
    def __init__(
            self, model: nn.Module, optimizer_params: typing.Optional[Box], id: int,
            train_loader: typing.Optional[DataLoader] = None,
            valid_loader: typing.Optional[DataLoader] = None,
            result_dir: typing.Optional[str] = None, log_every=100,
            model_saving_strategy=LearnerModelSavingStrategy.NEVER,
            evaluation_strategy=LearnerEvaluationStrategy.NEVER, scale: float = 0.0
            ):

        super().__init__(
            model, optimizer_params, id, train_loader, valid_loader, result_dir, log_every,
            model_saving_strategy, evaluation_strategy
            )
        self.gradients = []
        self.scale = scale

    def before_weight_update(self):
        for idx, (name, _) in enumerate(self.model.named_parameters()):
            grads = torch.stack([g[idx] for g in self.gradients], dim=0)
            new_grads = project_gradients(grads)
            unique = grads - new_grads

            # correct logic
            new_grads += self.scale * unique
            dict(self.model.named_parameters())[name].grad = new_grads.mean(dim=0)

    def train(self, *, num_epochs, epoch_offset, round, device="cpu", **kwargs):
        logger.info(f"Starting to train learner: {self.id}")
        self.on_train_begin(
            num_epochs=num_epochs, epoch_offset=epoch_offset, round=round, device=device, **kwargs
            )

        for epoch in range(epoch_offset, epoch_offset + num_epochs):
            self.on_epoch_begin(epoch=epoch, **kwargs)

            # train loop
            self.model.train()
            for i, batch in enumerate(self.train_loader):
                pred = self.model(batch)
                loss, aux_loss = self.model.loss(pred, batch, reduce=False)

                # compute gradients
                # noinspection PyAttributeOutsideInit
                self.gradients = []
                for idx, l in enumerate(loss):
                    g = torch.autograd.grad(l, list(self.model.parameters()), retain_graph=True)
                    self.gradients.append(g)

                # modify gradients
                self.before_weight_update()

                self.optimizer.step()

                # clear gradients
                self.optimizer.zero_grad()
                self.gradients = []

                loss = loss.mean()
                for (k, v) in aux_loss.items():
                    aux_loss[k] = aux_loss[k].mean()

                self.step += 1

                loss_logger_helper(
                    loss, aux_loss, writer=self.summary_writer,
                    step=self.step, epoch=epoch,
                    log_every=self.log_every, string="train"
                    )

            self.on_epoch_end(epoch=epoch, **kwargs)

        self.on_train_end(
            num_epochs=num_epochs, epoch_offset=epoch_offset, round=round, **kwargs
            )
        logger.info(f"Finished training learner: {self.id}")
