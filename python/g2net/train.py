import pandas as pd
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from typing import Tuple, List
from catalyst import dl
from catalyst.callbacks.checkpoint import CheckpointCallback

from g2net.io.dataset import G2NetDataset
from g2net.io.transforms import Compose, Normalize, BandPass, GaussianNoiseSNR
from g2net.models.filter import MiniRocket
from g2net.utils.tsai import Timer


class TrainPipeline(object):
    """Basic pipeline for training the models.

    It's generally you feed the params into this class with a dict and do:
        TrainPipeline(**params_kwargs)
    """

    def __init__(self,
                 train_loader: DataLoader,
                 valid_loader: DataLoader,
                 lr: float = 2e-4,
                 num_epochs: int = 15,
                 model_params: dict = None,
                 scheduler_params: dict = None,
                 save_path: str = "model.pt") -> None:
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.lr = lr
        self.num_epochs = num_epochs
        self.model = None
        self.model_params = model_params
        if scheduler_params == None:
            self.scheduler_params = {"T_0": 5, "T_mult": 1, "eta_min": 1e-6}
        else:
            self.scheduler_params = scheduler_params

        self.save_path = save_path

    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)

    def get_model_info(self,
                       input_shape: Tuple[str] = (3, 4096),
                       cuda: bool = True):
        if self.model == None:
            print("No model found")
        else:
            from torchsummary import summary
            if cuda:
                self.model = self.model.to("cuda")
            summary(self.model, input_shape)

    def create_baseline(self):
        from g2net.models.base.architectures import SpectroCNN
        from g2net.models.base.wavegram import CNNSpectrogram

        spec_params = dict(
            base_filters=128,
            kernel_sizes=(64, 16, 4),
        )

        model_params = dict(model_name="tf_efficientnet_b6_ns",
                            pretrained=True,
                            spectrogram=CNNSpectrogram,
                            spec_params=spec_params,
                            custom_classifier="gem",
                            upsample="bicubic")
        self.model = SpectroCNN(**model_params)

    def create_minirocket(self):
        # Online mode; head is learned
        print("Creating minirocket model...")
        if self.model_params == None:
            self.model_params = {
                "c_in": 3,
                "c_out": 1,
                "seq_len": 4096,
                "num_features": 10000,
                "random_state": 2021
            }

        self.model = MiniRocket(**self.model_params)

    def train_minirocket(self):
        """Simple pipeline with minimal configuration.
        """
        self.create_minirocket()
        self.get_model_info(input_shape=(3, 4096))

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        loaders = {
            "train": self.train_loader,
            "valid": self.valid_loader,
        }
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, **self.scheduler_params)

        runner = dl.SupervisedRunner(input_key="features",
                                     output_key="logits",
                                     target_key="targets",
                                     loss_key="loss")
        timer = Timer()
        timer.start()
        # model training
        runner.train(
            model=self.model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            loaders=loaders,
            num_epochs=self.num_epochs,
            callbacks=[
                CheckpointCallback(use_runner_logdir=True),
                dl.EarlyStoppingCallback(patience=2,
                                         loader_key="valid",
                                         metric_key="loss",
                                         minimize=True),
                dl.BatchTransformCallback(transform=torch.sigmoid,
                                          scope="on_batch_end",
                                          input_key="logits",
                                          output_key="scores"),
                dl.AUCCallback(input_key="scores", target_key="targets"),
            ],
            logdir="./logs",
            valid_loader="valid",
            valid_metric="loss",
            minimize_valid_metric=True,
            verbose=True,
            load_best_on_end=True,
        )
        timer.stop()
        print(f"Saving {self.save_path}...")
        self.save_model(self.save_path)
        return self.model


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
