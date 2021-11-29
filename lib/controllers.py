import copy
import logging
import math
import typing
from enum import Enum
from multiprocessing import Value
from time import sleep

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib import os_utils
from lib.learners import Learner
from lib.os_utils import loss_logger_helper

logger = logging.getLogger()


class ControllerEvaluationStrategy(Enum):
    NEVER = "never"
    EVERY_ROUND = "per_round"
    END_OF_TRAINING = "end_of_training"


def add_params(combined_params, param):
    for key in combined_params:
        combined_params[key] += param[key]
    return combined_params


class ControllerModelSavingStrategy(Enum):
    NEVER = "never"
    EVERY_ROUND = "per_round"
    END_OF_TRAINING = "end_of_training"


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
