import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from typing import List
from sklearn.metrics import roc_auc_score

from g2net.io.dataset import G2NetDataset
from g2net.io.transforms import Compose
from g2net.models.filter import MiniRocket
from g2net.models.base import create_base_model
from g2net.train import create_base_transforms
from g2net.utils.tsai import Timer


class Inferer(object):
    """Runs and collects stats for the inference. Single GPU inference.

    The associated test set should be on df.iloc[dset_size:] (not covered by the
    train and validation folds).

    Should log:
    - Time each run takes
    - The trial/fold
    - The performance on the test sets
    - Standard deviation and averages for all trials

    Should be able to sample a distribution:
    - 1% are GW
    - 10% are GW
    - 50% are GW
        - what is the predicted distributions for each type?
    """

    def __init__(self,
                 test_loader,
                 base_model_paths: List[str],
                 filter_model_paths: List[str],
                 threshold: float = 0.5,
                 num_folds: int = 5):
        """
        Args:
            test_loader: See create_test_dataloader
            base_model_paths: paths to base models in order of folds
            filter_model_paths: paths to filter models in order of folds
            threshold: the cutoff for sigmoid predicted outputs for 0 and 1.
            num_folds: Should be = len(base_model_paths)
        """
        self.test_loader = test_loader
        self.base_model_paths = base_model_paths
        self.filter_model_paths = filter_model_paths
        self.timer = Timer()
        self.base_model = None
        self.filter_model = None
        self.threshold = threshold
        self.num_folds = num_folds

        if self.num_folds != len(self.base_model_paths):
            raise ValueError(
                "the number of folds should be the same as the number of base model weights"
            )

        if self.num_folds != len(self.filter_model_paths):
            raise ValueError(
                "the number of folds should be the same as the number of base model weights"
            )

        cols = [
            "fold", "base_auc", "base_time", "mr_auc", "mr_time", "both_auc",
            "both_time"
        ]
        print("Tracking ", cols)
        # To get averages, do self.metrics.mean()
        self.metrics = pd.DataFrame(columns=cols)

    def infer_single(self, model, metric_key):
        if model == None:
            raise ValueError("model not defined")

        self.timer.start(verbose=False)
        all_auc = []
        # Single GPU inference
        for batch in self.test_loader:
            pred = model(batch[0])
            # thresholding
            pred[pred >= self.threshold] = 1
            pred[pred < self.threshold] = 0
            # evaluate
            auc = infer_auc(batch[1], pred)
            all_auc.append(auc)
        time_elapsed = self.timer.stop()

        return {
            f'{metric_key}_auc': np.average(all_auc),
            f'{metric_key}_time': time_elapsed,
        }

    def infer_minirocket_only(self, fold) -> dict:
        """Infers minirocket only. Returns a dictionary of metrics
        """
        # loads model and weights
        self.filter_model = MiniRocket()
        load_weights(self.filter_model, self.filter_model_paths[fold])
        self.filter_model.eval()

        return self.infer_single(self.filter_model, "mr")

    def infer_base_only(self, fold) -> dict:
        """Infers with only the base model. Returns a dictionary of metrics
        """
        # loads model and weights
        self.base_model = create_base_model()
        load_weights(self.base_model, self.base_model_paths[fold])
        self.base_model.eval()

        # Do inference
        return self.infer_single(self.base_model, "base")

    def infer_both(self, filter_model: torch.nn.Module,
                   base_model: torch.nn.Module):
        """Inference that works with a general filter_model and base_model.
        Filter_model is recommended to be a torch model. Should be able to
        invoke filter_model(x) to generate batch.
        """
        self.timer.start(verbose=False)
        all_auc = []

        # Single GPU inference
        for batch in self.test_loader:
            pred = filter_model(batch[0])
            # indices of the batch that the model thought were GWs
            pos_idx = pred >= self.threshold
            if len(pos_idx) > 0:
                pred[pos_idx] = base_model(batch[0][pos_idx])

            # thresholding with the new predictions (if any)
            pred[pred >= self.threshold] = 1
            pred[pred < self.threshold] = 0
            # evaluate
            auc = infer_auc(batch[1], pred)
            all_auc.append(auc)

        time_elapsed = self.timer.stop()

        return {
            f'both_auc': np.average(all_auc),
            f'both_time': time_elapsed,
        }

    def infer_single_fold(self, fold: int):
        """Explicit inference with minirocket as the filter model and the
        baseline solution.
        """
        # TO DO: Inference with class sampling
        # Load models and weights
        mr_only_metrics = self.infer_minirocket_only(fold)
        base_only_metrics = self.infer_base_only(fold)
        both_metrics = self.infer_all_both(self.filter_model, self.base_model)
        fold_metrics = {**mr_only_metrics, **base_only_metrics, **both_metrics}
        return fold_metrics

    def infer_all(self):
        """Inference on all folds
        """
        for fold in range(self.num_folds):
            metrics = self.infer_single_fold(fold)
            self.metrics.append({"fold": fold, **metrics})
        # TO DO: compute averages and std dev and append the row


def infer_auc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Wraps sklearn.metrics.roc_auc_score to work with torch tensors.
    """
    y = y_true.cpu().numpy().flatten()
    pred = y_pred.cpu().numpy().flatten()
    return roc_auc_score(y, pred)


def load_weights(model: torch.nn.Module, weights_path: str):
    """Loads weights into model from weights_path.

    The state dict from the baseline has the keys:
        "global_epoch", "model", "state", "all_states"
    To load the model weights, only use state_dict["model"].
    """
    model.load_state_dict(torch.load(weights_path)["model"])


def create_test_transforms(maxes: List[float] = [4.61e-20, 4.23e-20, 1.11e-20],
                           bandpass_range: List[int] = [16, 512]) -> Compose:
    return create_base_transforms(maxes, bandpass_range=bandpass_range)["test"]


def create_test_dataloader(test_df: pd.DataFrame,
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