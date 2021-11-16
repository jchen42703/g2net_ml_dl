import datetime
from catalyst import dl
from g2net.models.filter import MiniRocket
from g2net.utils.tsai import Timer
from torch import nn, optim
import torch


class TrainPipeline(object):
    """Basic pipeline for training the models.
    """

    def __init__(
        self,
        train_loader: torch.data.DataLoader,
        valid_loader: torch.data.DataLoader,
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
