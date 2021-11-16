import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List
import datetime
from catalyst import dl

from g2net.io.dataset import G2NetDataset
from g2net.io.transforms import Compose, Normalize, BandPass, GaussianNoiseSNR
from g2net.models.filter import MiniRocket
from g2net.utils.tsai import Timer


class TrainPipeline(object):
    """Basic pipeline for training the models.
    """

    def __init__(
        self,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        lr: float = 1e-2,
        num_epoch: int = 100,
    ) -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr = lr
        self.num_epoch = num_epoch

    def save_model(self, model: torch.nn.Module, path: str):
        torch.save(model.state_dict(), path)

    def train_minirocket(self):
        # Online mode; head is learned
        model = MiniRocket(3, 1, 4096, num_features=50000, random_state=420)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)

        loaders = {
            "train": self.train_loader,
            "valid": self.valid_loader,
        }

        runner = dl.SupervisedRunner(input_key="features",
                                     output_key="logits",
                                     target_key="targets",
                                     loss_key="loss")
        timer = Timer()
        timer.start()
        # model training
        runner.train(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            loaders=loaders,
            num_epochs=self.num_epoch,
            callbacks=[
                dl.AccuracyCallback(input_key="logits", target_key="targets"),
                dl.AUCCallback(input_key="logits", target_key="targets"),
            ],
            logdir="./logs",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
            load_best_on_end=True,
        )
        timer.stop()
        timestamp = datetime.datetime.now().timestamp()
        self.save_model(f'minirocket_{timestamp}.pt')
        return model


def create_dataloaders(train_fold: pd.DataFrame,
                       valid_fold: pd.DataFrame,
                       batch_size: int = 64,
                       train_transforms: List = None,
                       test_transforms: List = None) -> Tuple[DataLoader]:
    train_dset = G2NetDataset(paths=train_fold['path'].values,
                              targets=train_fold['target'].values,
                              transforms=train_transforms,
                              is_test=False)
    valid_dset = G2NetDataset(paths=valid_fold['path'].values,
                              targets=valid_fold['target'].values,
                              transforms=test_transforms,
                              is_test=True)
    train_loader = DataLoader(train_dset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=False)
    valid_loader = DataLoader(valid_dset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=0,
                              pin_memory=False)
    return (train_loader, valid_loader)


def sota_dl_transforms() -> dict:
    """Creates a dictionary with keys train, test, and tta with the necessary
    transforms for the config that we chose from the 2nd place sol.

    The only transforms are normalization, a bandpass filter and gauss noise.
    """
    transforms = dict(
        train=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
        tta=Compose([
            Normalize(factors=[4.61e-20, 4.23e-20, 1.11e-20]),
            BandPass(lower=16, upper=512),
        ]),
    )

    return transforms


def create_base_transforms(maxes: List[float] = [4.61e-20, 4.23e-20, 1.11e-20],
                           bandpass_range: List[int] = [16, 512]) -> dict:
    """Creates a dictionary with keys train, test, and tta with the necessary
    transforms for the config that we chose from the 2nd place sol.

    The only transforms are normalization, a bandpass filter and gauss noise.
    """
    transforms = dict(
        train=Compose([
            Normalize(factors=maxes),
            BandPass(lower=bandpass_range[0], upper=bandpass_range[1]),
            GaussianNoiseSNR(min_snr=15, max_snr=30, p=0.5),
        ]),
        test=Compose([
            Normalize(factors=maxes),
            BandPass(lower=bandpass_range[0], upper=bandpass_range[1]),
        ]),
        tta=Compose([
            Normalize(factors=maxes),
            BandPass(lower=bandpass_range[0], upper=bandpass_range[1]),
        ]),
    )

    return transforms
