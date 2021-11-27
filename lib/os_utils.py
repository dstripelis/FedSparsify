""" general utility functions"""
import argparse
import importlib
import json
import logging
import random
import re
import shutil
import sys
import typing
from argparse import ArgumentParser
from collections.abc import MutableMapping

import numpy
import torch
from box import Box

logger = logging.getLogger()


def listorstr(inp):
    if len(inp) == 1:
        return try_cast(inp[0])

    for i, val in enumerate(inp):
        inp[i] = try_cast(val)
    return inp


def try_cast(text):
    """ try to cast to int or float if possible, else return the text itself"""
    result = try_int(text, None)
    if result is not None:
        return result

    result = try_float(text, None)
    if result is not None:
        return result

    return text


def try_float(text, default: typing.Optional[int] = 0.0):
    result = default
    try:
        result = float(text)
    except Exception as _:
        pass
    return result


def try_int(text, default: typing.Optional[int] = 0):
    result = default
    try:
        result = int(text)
    except Exception as _:
        pass
    return result


def parse_args(parser: ArgumentParser) -> Box:
    # get defaults
    defaults = {}
    # taken from parser_known_args code
    # add any action defaults that aren't present
    for action in parser._actions:
        if action.dest is not argparse.SUPPRESS:
            if action.default is not argparse.SUPPRESS:
                defaults[action.dest] = action.default

    # add any parser defaults that aren't present
    for dest in parser._defaults:
        defaults[dest] = parser._defaults[dest]

    # check if there is config & read config
    args = parser.parse_args()
    if vars(args).get("config") is not None:
        # load a .py config
        configFile = args.config
        spec = importlib.util.spec_from_file_location("config", configFile)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = module.config
        # merge config and override defaults
        defaults.update({k: v for k, v in config.items()})

    # override defaults with command line params
    # this will get rid of defaults and only read command line args
    parser._defaults = {}
    parser._actions = {}
    args = parser.parse_args()
    defaults.update({k: v for k, v in vars(args).items()})

    return boxify_dict(defaults)


def boxify_dict(config):
    """
  this takes a flat dictionary and break it into sub-dictionaries based on "." seperation
    a = {"model.a":  1, "model.b" : 2,  "alpha" : 3} will return Box({"model" : {"a" :1,
    "b" : 2}, alpha:3})
    a = {"model.a":  1, "model.b" : 2,  "model" : 3} will throw error
  """
    new_config = {}
    # iterate over keys and split on "."
    for key in config:
        if "." in key:
            temp_config = new_config
            for k in key.split(".")[:-1]:
                # create non-existent keys as dictionary recursively
                if temp_config.get(k) is None:
                    temp_config[k] = {}
                elif not isinstance(temp_config.get(k), dict):
                    raise TypeError(f"Key '{k}' has values as well as child")
                temp_config = temp_config[k]
            temp_config[key.split(".")[-1]] = config[key]
        else:
            if new_config.get(key) is None:
                new_config[key] = config[key]
            else:
                raise TypeError(f"Key '{key}' has values as well as child")

    return Box(new_config)


# https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
def flatten(d, parent_key='', sep='.'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return Box(dict(items))


def str2bool(v: typing.Union[bool, str, int]) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1", 1):
        return True
    if v.lower() in ("no", "false", "f", "n", "0", 0):
        return False
    raise TypeError("Boolean value expected.")


def safe_isdir(dir_name):
    return os.path.exists(dir_name) and os.path.isdir(dir_name)


def safe_makedirs(dir_name):
    try:
        os.makedirs(dir_name)
    except OSError as e:
        print(e)


def jsonize(x: object) -> typing.Union[str, dict]:
    try:
        temp = json.dumps(x)
        return temp
    except Exception as e:
        return {}


def copy_code(folder_to_copy, out_folder, replace=False):
    logger.info(f"copying {folder_to_copy} to {out_folder}")

    if os.path.exists(out_folder):
        if not os.path.isdir(out_folder):
            logger.error(f"{out_folder} is not a directory")
            sys.exit()
        else:
            logger.info(f"Not deleting existing result folder: {out_folder}")
    else:
        os.makedirs(out_folder)

    # replace / with _
    folder_name = f'{out_folder}/{re.sub("/", "_", folder_to_copy)}'

    # create a new copy if something already exists
    if not replace:
        i = 1
        temp = folder_name
        while os.path.exists(temp):
            temp = f"{folder_name}_{i}"
            i += 1
        folder_name = temp
    else:
        if os.path.exists(folder_name):
            if os.path.isdir(folder_name):
                shutil.rmtree(folder_name)
            else:
                raise FileExistsError(
                    "There is a file with same name as folder")

    logger.info(f"Copying {folder_to_copy} to {folder_name}")
    shutil.copytree(folder_to_copy, folder_name)


def set_seed(seed):
    if isinstance(seed, list):
        torch_seed, numpy_seed, random_seed = seed
    else:
        torch_seed, numpy_seed, random_seed = seed, seed, seed

    torch.manual_seed(torch_seed)
    numpy.random.seed(numpy_seed)
    random.seed(random_seed)
