import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List

from g2net.io.dataset import G2NetDataset
from g2net.io.transforms import Compose
from g2net.models.filter import MiniRocket
from g2net.utils.tsai import Timer
from g2net.train import create_base_transforms


class Inferer(object):
    """Runs and collects stats for the inference.

    The associated test set should be on df.iloc[dset_size:] (not covered by the
    train and validation folds).

    Should log:
    - Time each run takes
    - The trial and fold
    - The performance on the test sets
    - Standard deviation and averages for all trials
    - GPU memory...?

    Should be able to sample a distribution:
    - 1% are GW
    - 10% are GW
    - 50% are GW
        - what is the predicted distributions for each type?
    """

    def __init__(self, test_loader):
        self.test_loader = test_loader
        self.timer = Timer()

    # def setup_pipeline(self):
    #     """Creates test dataloaders and
    #     """
    #     pass

    def infer_minirocket_only(self):
        pass

    def infer_base_only(self):
        pass

    def infer_both(self):
        pass


def create_test_transforms(maxes: List[float] = [4.61e-20, 4.23e-20, 1.11e-20],
                           bandpass_range: List[int] = [16, 512]) -> Compose:
    return create_base_transforms(maxes, bandpass_range=bandpass_range)["test"]


def create_infer_dataloaders(test_df: pd.DataFrame,
                             batch_size: int = 64,
                             test_transforms: List = None,
                             num_workers: int = 8) -> DataLoader:
    test_dset = G2NetDataset(paths=test_df['path'].values,
                             targets=test_df['target'].values,
                             transforms=test_transforms,
                             is_test=True)
    test_loader = DataLoader(test_dset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)
    return test_loader