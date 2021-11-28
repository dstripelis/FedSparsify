import copy
import logging
import math
import typing
from dataclasses import dataclass
from enum import Enum
from multiprocessing import Value
from time import sleep

import dill
import numpy
import torch
from box import Box
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib import os_utils

logger = logging.getLogger()


class LearnerModelSavingStrategy(Enum):
    NEVER = "never"
    EVERY_ROUND = "per_round"
    EVERY_EPOCH = "per_epoch"


class LearnerEvaluationStrategy(Enum):
    NEVER = "never"
    EVERY_ROUND = "per_round"
    EVERY_EPOCH = "per_epoch"


class ControllerEvaluationStrategy(Enum):
    NEVER = "never"
    EVERY_ROUND = "per_round"
    END_OF_TRAINING = "end_of_training"


class ControllerModelSavingStrategy(Enum):
    NEVER = "never"
    EVERY_ROUND = "per_round"
    END_OF_TRAINING = "end_of_training"


def loss_logger_helper(
        loss, aux_loss, writer: typing.Optional[SummaryWriter], step: int, epoch: int,
        log_every: int, string: str = "train", force_print: bool = False, new_line: bool = False
        ):
    # write to tensorboard at every step but only print at log step or when force_print is passed
    if writer is not None:
        writer.add_scalar(f"{string}/loss", loss, step)
        for k, v in aux_loss.items():
            writer.add_scalar(f"{string}/" + k, v, step)

    if step % log_every == 0 or force_print:
        logger.info(f"{string}/loss: ({step}/{epoch}) {loss}")

    if force_print:
        if new_line:
            for k, v in aux_loss.items():
                logger.info(f"{string}/{k}:{v} ")
        else:
            str_ = ""
            for k, v in aux_loss.items():
                str_ += f"{string}/{k}:{v} "
            logger.info(f"{str_}")


def add_params(combined_params, param):
    for key in combined_params:
        combined_params[key] += param[key]
    return combined_params


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
    evaluation_strategy: LearnerEvaluationStrategy = LearnerEvaluationStrategy.NEVER

    def create_learner(self):
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


class FedController:

    def __init__(
            self,
            learners: typing.List[Learner],
            device: str,
            test_loader=typing.Optional[DataLoader],
            result_dir=typing.Optional[str],
            model_saving_strategy=ControllerModelSavingStrategy.NEVER,
            evaluation_strategy=ControllerEvaluationStrategy.NEVER, ):

        logger.debug(f"Created Federation Controller with {len(learners)}")
        self.result_dir = result_dir
        os_utils.safe_makedirs(result_dir)

        # logging stuff
        if result_dir is not None:
            self.summary_writer = SummaryWriter(log_dir=result_dir)

        self.model_saving_strategy = model_saving_strategy
        self.evaluation_strategy = evaluation_strategy

        self.learners = learners
        self.test_loader = test_loader

        # create a handicapped learner instance for local evaluation
        self.community_evaluator = Learner(
            copy.deepcopy(learners[0].model), None, 0
            )
        self.device = device

    def train(self, *, num_rounds, epochs_per_round, **kwargs):

        logger.info("Starting federation controller")
        self.on_train_begin(
            num_rounds=num_rounds,
            epochs_per_round=epochs_per_round, **kwargs
            )

        for round in range(num_rounds):

            logger.info(f"Starting round {round}")
            self.on_round_begin(
                round=round, num_rounds=num_rounds,
                epochs_per_round=epochs_per_round, **kwargs
                )

            for learner in self.learners:
                logger.info(f"Round {round}: Training learner {learner.id}")
                self.on_learner_train_begin(
                    round=round, num_rounds=num_rounds,
                    learner_id=learner.id,
                    epochs_per_round=epochs_per_round,
                    **kwargs
                    )

                learner.train(
                    num_epochs=epochs_per_round, device=self.device,
                    epoch_offset=round * epochs_per_round,
                    round=round,
                    **kwargs
                    )

                self.on_learner_train_end(
                    round=round, learner_id=learner.id,
                    num_rounds=num_rounds,
                    epochs_per_round=epochs_per_round,
                    **kwargs
                    )
                logger.info(
                    f"Round {round}: Finished Training learner {learner.id}"
                    )

            logger.info(f"Finished training learners in round {round}")

            logger.info("Computing community updates")
            community_params = self.combine_learners()

            logger.info("Applying community updates")
            [learner.set_params(community_params) for learner in self.learners]
            self.community_evaluator.set_params(community_params)
            self.on_round_end(
                round=round, num_rounds=num_rounds,
                epochs_per_round=epochs_per_round, **kwargs
                )
            logger.info(f"Round {round} finished")

        self.on_train_end(
            num_rounds=num_rounds,
            epochs_per_round=epochs_per_round, **kwargs
            )

    def combine_learners(self):
        combined_params = self.learners[0].get_params()
        for i in range(1, len(self.learners)):
            combined_params = add_params(
                combined_params,
                self.learners[i].get_params()
                )

        num_learner = len(self.learners)
        for key in combined_params:
            combined_params[key] /= num_learner

        return combined_params

    # declare hook functions
    def on_train_begin(self, **kwargs):
        logger.debug("Controller:In train begin")
        pass

    def on_train_end(self, *, num_rounds, **kwargs):
        logger.debug("Controller:In train end")

        # save params
        if self.model_saving_strategy == ControllerModelSavingStrategy.END_OF_TRAINING:
            self.learners[0].save(f"{self.result_dir}/community_params_final.pt")

        # do evaluation
        if self.evaluation_strategy == ControllerEvaluationStrategy.END_OF_TRAINING:
            if self.test_loader:
                loss, aux_loss = self.evaluate(self.test_loader)
                loss_logger_helper(
                    loss, aux_loss, writer=self.summary_writer,
                    step=num_rounds, epoch=num_rounds,
                    log_every=100,
                    string="controller",
                    force_print=True
                    )
            else:
                logger.info(
                    f"Didnot evaluate community params at the end of training because data loader not found"
                    )

    def on_round_begin(self, **kwargs):
        logger.debug("Controller:In round begin")

        pass

    def on_round_end(self, *, round, **kwargs):
        logger.debug("Controller:In round end")

        # save params
        if self.model_saving_strategy == ControllerModelSavingStrategy.EVERY_ROUND:
            self.learners[0].save(f"{self.result_dir}/community_params_round_{round}.pt")

        # do evaluation
        if self.evaluation_strategy == ControllerEvaluationStrategy.EVERY_ROUND:
            if self.test_loader:
                loss, aux_loss = self.evaluate(self.test_loader)
                loss_logger_helper(
                    loss, aux_loss, writer=self.summary_writer,
                    step=round, epoch=round, log_every=100,
                    string="controller",
                    force_print=True
                    )
            else:
                logger.info(
                    f"Didnot evaluate community params at the end of round because data loader not found"
                    )

    def on_learner_train_begin(self, **kwargs):
        logger.debug("Controller:In learner train begin")
        pass

    def on_learner_train_end(self, **kwargs):
        logger.debug("Controller:In learner train end")
        pass

    def evaluate(self, loader, **kwargs):
        self.community_evaluator.model.to(self.device)
        result = self.community_evaluator.evaluate(loader)
        self.community_evaluator.model.to("cpu")
        return result


# TODO:
#  - SummaryWriter is not picklable which gives the thread._lock error. Find an alternate to that
#  - Logger doesnot work properly, use this solution https://stackoverflow.com/questions/16933888/logging-while-using-parallel-python
#  - the target argument to process has to be a global function, not attached to class. 
class ParallelFedController(FedController):

    def __init__(
            self, learners: typing.List[Learner], device: str,
            test_loader=typing.Optional[DataLoader], result_dir=typing.Optional[str],
            model_saving_strategy=ControllerModelSavingStrategy.NEVER,
            evaluation_strategy=ControllerEvaluationStrategy.NEVER,
            devices: typing.Optional[typing.List] = None,
            max_process_per_device: int = 1,
            ):

        super().__init__(
            learners, device, test_loader, result_dir, model_saving_strategy, evaluation_strategy
            )
        if devices is None:
            devices = []
        self.devices = devices
        self.max_process_per_device = max_process_per_device

    def train(self, *, num_rounds, epochs_per_round, **kwargs):
        from multiprocessing import Process

        logger.info("Starting federation controller")
        self.on_train_begin(
            num_rounds=num_rounds,
            epochs_per_round=epochs_per_round, **kwargs
            )
        # breakpoint()

        for round in range(num_rounds):

            logger.info(f"Starting round {round}")
            self.on_round_begin(
                round=round, num_rounds=num_rounds,
                epochs_per_round=epochs_per_round, **kwargs
                )
            devices = self.devices * math.ceil(len(self.learners) / len(self.devices))

            print(devices)
            self.count = Value("i", 0)
            processes = [Process(
                target=_learner_training, kwargs={
                    "learner"         : l,
                    "round"           : round,
                    "num_rounds"      : num_rounds,
                    "epochs_per_round": epochs_per_round,
                    "device"          : device,
                    "counter"         : self.count
                    }
                ) for (l, device) in zip(self.learners, devices)]
            idx = 0
            max_process_count = self.max_process_per_device * len(self.devices)
            while idx < len(processes):
                if self.count.value < max_process_count:
                    with self.count.get_lock():
                        self.count.value += 1
                    processes[idx].start()
                    idx += 1
                    print(idx)
                else:
                    sleep(10)
            for p in processes:
                logger.info("waiting")
                p.join()

            logger.info(f"Finished training learners in round {round}")

            logger.info("Computing community updates")
            community_params = self.combine_learners()

            logger.info("Applying community updates")
            [learner.set_params(community_params) for learner in self.learners]
            self.community_evaluator.set_params(community_params)
            self.on_round_end(
                round=round, num_rounds=num_rounds,
                epochs_per_round=epochs_per_round, **kwargs
                )
            logger.info(f"Round {round} finished")

        self.on_train_end(
            num_rounds=num_rounds,
            epochs_per_round=epochs_per_round, **kwargs
            )


def _learner_training(
        learner, round, num_rounds, epochs_per_round, device, counter, **kwargs
        ):
    logger.info(f"Round {round}: Training learner {learner.id}")
    # self.on_learner_train_begin(
    #     round=round, num_rounds=num_rounds, learner_id=learner.id,
    #     epochs_per_round=epochs_per_round, **kwargs
    #     )

    learner.train(
        num_epochs=epochs_per_round, device=device, epoch_offset=round * epochs_per_round,
        round=round, **kwargs
        )

    # self.on_learner_train_end(
    #     round=round, learner_id=learner.id, num_rounds=num_rounds,
    #     epochs_per_round=epochs_per_round, **kwargs
    #     )
    logger.info(
        f"Round {round}: Finished Training learner {learner.id}"
        )
    with counter.get_lock():
        counter.value -= 1
