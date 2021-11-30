import argparse
import importlib
import json
import logging
import pprint

import torch
from box import Box
from torch.utils.data import DataLoader

from lib import os_utils
from lib.controllers import ControllerEvaluationStrategy, ControllerModelSavingStrategy, \
    FedController, ParallelFedController
from lib.data.datasets import get_dataset
from lib.learners import LearnerEvaluationStrategy, LearnerMetaData, LearnerModelSavingStrategy

from lib.models.regression import Regression


def parser_setup():
    # define argparsers
    str2bool = os_utils.str2bool
    listorstr = os_utils.listorstr

    parser = argparse.ArgumentParser()
    parser.add_argument("-D", "--debug", action='store_true')
    parser.add_argument("--config", "-c", required=False)
    parser.add_argument("--seed", required=False, type=int)

    parser.add_argument("--exp_name", required=True)

    parser.add_argument(
        "--device", required=False,
        default="cuda" if torch.cuda.is_available() else "cpu"
        )
    parser.add_argument("--result_folder", "-r", required=False)
    parser.add_argument(
        "--mode", required=False, nargs="+", choices=["test", "train"],
        default=["test", "train"]
        )
    parser.add_argument("--statefile", "-s", required=False, default=None)
    parser.add_argument("--use_parallel_controller", default=False, type=str2bool)

    # data related arguments
    parser.add_argument("--data.name", "-d", required=False, choices=["brain_age"])
    parser.add_argument("--data.root_path", default=None, type=str)
    parser.add_argument("--data.train_csv_folder", default=None, type=str)
    parser.add_argument("--data.valid_csv_path", default=None, type=str)
    parser.add_argument("--data.test_csv_path", default=None, type=str)
    parser.add_argument(
        "--data.frame_dim", default=1, type=int, choices=[1, 2, 3],
        help="choose which dimension we want to slice, 1 for sagittal, 2 for coronal, 3 for axial"
        )

    parser.add_argument("--model.name", required=False, choices=["regression"])

    parser.add_argument("--model.arch.file", required=False, type=str, default=None)
    parser.add_argument("--model.arch.attn_num_heads", required=False, type=int, default=1)
    parser.add_argument("--model.arch.attn_dim", required=False, type=int, default=128)
    parser.add_argument("--model.arch.attn_drop", required=False, type=str2bool, default=False)
    parser.add_argument(
        "--model.arch.agg_fn", required=False, type=str, choices=["mean", "max", "attention"]
        )

    parser.add_argument("--train.batch_size", required=False, type=int, default=8)
    parser.add_argument(
        "--train.optimizer", required=False, type=str, default="adam", choices=["adam", "sgd"]
        )
    parser.add_argument("--train.lr", required=False, type=float, default=1e-3)
    parser.add_argument("--train.weight_decay", required=False, type=float, default=0.0)
    parser.add_argument("--train.log_every", required=False, type=int, default=1000)
    parser.add_argument(
        "--train.learner_model_saving_strategy", required=False, default="never",
        type=str, choices=["never", "per_round", "per_epoch"]
        )
    parser.add_argument(
        "--train.learner_evaluation_strategy", required=False, default="never",
        type=str, choices=["never", "per_round", "per_epoch"]
        )

    parser.add_argument(
        "--train.community_model_saving_strategy", required=False, type=str,
        default="never", choices=["never", "per_round", "end_of_training"]
        )
    parser.add_argument(
        "--train.community_evaluation_strategy", required=False, type=str,
        default="never", choices=["never", "per_round", "end_of_training"]
        )

    parser.add_argument("--train.num_rounds", required=False, default=25, type=int)
    parser.add_argument("--train.epochs_per_round", required=False, default=4, type=int)

    parser.add_argument("--train.differential_privacy", default=False, type=str2bool)
    parser.add_argument("--train.virtual_batch_size", default=None, type=int)
    parser.add_argument("--train.noise_multiplier", default=1.0, type=float)
    parser.add_argument("--train.grad_norm", default=1.0, type=float)

    parser.add_argument("--train.unique_gradient_learner", default=False, type=str2bool)
    parser.add_argument("--train.unique_gradient_scale", default=0.0, type=float)

    parser.add_argument("--test.batch_size", required=False, type=int, default=8)
    return parser


def load_data(config):
    data, meta = get_dataset(
        name=config.data.name, root_path=config.data.root_path,
        train_csv_folder=config.data.train_csv_folder, valid_csv=config.data.valid_csv_path,
        test_csv=config.data.test_csv_path
        )

    num_workers = 8
    logger.info(f"Using {num_workers} workers")
    train_loaders = [
        DataLoader(
            i, shuffle=True, batch_size=config.train.batch_size, num_workers=num_workers
            ) for i in data["train"]
        ]
    valid_loaders, test_loader = [None for i in train_loaders], None
    if data["valid"]:
        valid_loaders = [
            DataLoader(
                i, shuffle=True, batch_size=config.train.batch_size, num_workers=num_workers
                ) for i in data["valid"]
            ]

    if data["test"]:
        test_loader = DataLoader(
            data["test"], shuffle=False, batch_size=config.test.batch_size, num_workers=num_workers
            )
    return train_loaders, valid_loaders, test_loader, meta


def load_model(config):
    # we separate architecture and "model".
    # "Model" takes care of setting up losses etc.
    # This way we can keep trying different architecture and
    # keep loss function or model specific issues at one place

    # load arch module; this is a dynamically imported file
    arch_module = importlib.import_module(config.model.arch.file.replace("/", ".")[:-3])
    model_arch = arch_module.get_arch(
        input_shape=metadata.get("input_shape"), output_size=metadata.get("num_class"),
        **config.model.arch, slice_dim=config.data.frame_dim
        )

    # declaring models
    if config.model.name in "regression":
        model = Regression(**model_arch)
    else:
        raise Exception("Unknown model")
    return model


if __name__ == "__main__":

    # set seeds etc here
    torch.backends.cudnn.benchmark = True
    torch.multiprocessing.set_start_method('spawn')

    # define logger etc
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    logger = logging.getLogger()

    parser = parser_setup()
    config = os_utils.parse_args(parser)
    if config.seed is not None:
        os_utils.set_seed(config.seed)
    if config.debug:
        logger.setLevel(logging.DEBUG)

    logger.info("Config:")
    logger.info(pprint.pformat(config.to_dict(), indent=4))
    os_utils.safe_makedirs(config.result_folder)
    json.dump(config.to_dict(), open(f"{config.result_folder}/config.json", "w"))

    # Load data and create dataloaders
    logger.info("Getting data and dataloaders")
    train_loaders, valid_loaders, test_loader, metadata = load_data(config)
    num_learners = len(train_loaders)

    # create model for each learner
    models = [load_model(config) for _ in range(num_learners)]

    # We make sure each model has same initialization
    # We "randomly" choose first model and copy it's params to every model
    state_dict = models[0].state_dict()
    for i in range(num_learners):
        models[i].load_state_dict(state_dict)

    # create learners
    optimizer_params = Box(
        {
            "lr"                : config.train.lr,
            "optimizer"         : config.train.optimizer,
            "weight_decay"      : config.train.weight_decay,
            "reset_optimizer"   : config.train.get("reset_optimizer", True),
            "private_training"  : config.train.get("differential_privacy", False),
            "virtual_batch_size": config.train.get("virtual_batch_size", config.train.batch_size),
            "noise_multiplier"  : config.train.noise_multiplier,
            "grad_norm"         : config.train.grad_norm,
            }
        )

    learners_metadata = []
    for idx, (m, t, v) in enumerate(zip(models, train_loaders, valid_loaders)):
        optimizer_params.update({"sample_size": len(t.dataset)})
        learners_metadata.append(
            LearnerMetaData(
                m, optimizer_params, id=idx + 1, train_loader=t, valid_loader=v,
                result_dir=f"{config.result_folder}/learner_{idx + 1}",
                log_every=config.train.log_every,
                model_saving_strategy=LearnerModelSavingStrategy(
                    config.train.learner_model_saving_strategy
                    ),
                evaluation_strategy=LearnerEvaluationStrategy(
                    config.train.learner_evaluation_strategy
                    ),
                unique_gradient_learner=config.train.unique_gradient_learner,
                scale=config.train.unique_gradient_scale,
                ),
            )

    if config.use_parallel_controller:
        fed_trainer = ParallelFedController(
            learners=[i.create_learner() for i in learners_metadata],
            device=config.device,
            test_loader=test_loader,
            result_dir=config.result_folder,
            model_saving_strategy=ControllerModelSavingStrategy(
                config.train.community_model_saving_strategy
                ),
            evaluation_strategy=ControllerEvaluationStrategy(
                config.train.community_evaluation_strategy
                ),
            devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"]  # hardcoded for now
            )

        fed_trainer.train(
            num_rounds=config.train.num_rounds, epochs_per_round=config.train.epochs_per_round
            )
    else:
        fed_trainer = FedController(
            learners=[i.create_learner() for i in learners_metadata],
            device=config.device,
            test_loader=test_loader,
            result_dir=config.result_folder,
            model_saving_strategy=ControllerModelSavingStrategy(
                config.train.community_model_saving_strategy
                ),
            evaluation_strategy=ControllerEvaluationStrategy(
                config.train.community_evaluation_strategy
                )
            )

        fed_trainer.train(
            num_rounds=config.train.num_rounds, epochs_per_round=config.train.epochs_per_round
            )