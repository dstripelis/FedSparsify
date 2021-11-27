import logging
import os

from box import Box

from lib.data.ukbb_brain_age import UKBBBrainAGE

logger = logging.getLogger()


def get_dataset(name, train_csv_folder, valid_csv=None, test_csv=None,
                root_path=None):
    """ return dataset """
    if name == "brain_age":
        train_data = []
        test_data = None
        valid_data = None
        for file in sorted(os.listdir(train_csv_folder)):
            train_csv = f"{train_csv_folder}/{file}"
            train_data.append(
                UKBBBrainAGE(root=root_path, metadatafile=train_csv))

        if test_csv:
            test_data = UKBBBrainAGE(root=root_path, metadatafile=test_csv)
        if valid_csv:
            # use same valid data for each learner
            valid_data = [UKBBBrainAGE(root=root_path, metadatafile=valid_csv)
                          for _ in train_data]

        data = Box(
            {"train": train_data, "test": test_data, "valid": valid_data})
        metadata = {}
        return data, metadata

    logger.error(f"Invalid data name {name} specified")
    raise Exception(f"Invalid data name {name} specified")
