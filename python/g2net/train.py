import pandas as pd
from torch.utils.data import DataLoader
from g2net.io.dataset import G2NetDataset
from typing import Tuple, List
from g2net.io.transforms import Compose, Normalize, BandPass, GaussianNoiseSNR


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