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


class Inferrer(object):
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
                 filter_model_params: dict = None,
                 cpu_only: bool = False):
        """
        Args:
            test_loader: See create_test_dataloader
            base_model_paths: paths to base models in order of folds
            filter_model_paths: paths to filter models in order of folds
            threshold: the cutoff for sigmoid predicted outputs for 0 and 1.
        """
        self.test_loader = test_loader
        self.base_model_paths = base_model_paths
        self.filter_model_paths = filter_model_paths
        self.timer = Timer()
        self.base_model = None
        self.filter_model = None
        self.threshold = threshold
        self.num_folds = len(self.filter_model_paths)
        self.filter_model_params = filter_model_params
        self.cpu_only = cpu_only

        if len(self.filter_model_paths) != len(self.base_model_paths):
            raise ValueError(
                "the number of filter models should be the same as the number of base model weights"
            )

        self.cols = [
            "fold", "base_auc", "base_time", "mr_auc", "mr_time", "both_auc",
            "both_time"
        ]
        print("Tracking ", self.cols)
        # To get averages, do self.metrics.mean()
        self.metrics = pd.DataFrame(columns=self.cols)

    def infer_single(self, model, metric_key):
        if model == None:
            raise ValueError("model not defined")

        self.timer.start(verbose=False)
        all_auc = []
        # Single GPU inference
        for batch in self.test_loader:
            pred = predict_binary(model, batch[0])
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
        # Online mode; head is learned
        # print("Creating minirocket model...")
        if self.filter_model_params == None:
            print("Loading default minirocket params...")
            self.filter_model_params = {
                "c_in": 3,
                "c_out": 1,
                "seq_len": 4096,
                "num_features": 10000,
                "random_state": 2021
            }

        self.filter_model = MiniRocket(**self.filter_model_params)
        load_weights(self.filter_model,
                     self.filter_model_paths[fold],
                     cpu_only=self.cpu_only)
        self.filter_model.eval()

        return self.infer_single(self.filter_model, "mr")

    def infer_base_only(self, fold) -> dict:
        """Infers with only the base model. Returns a dictionary of metrics
        """
        # loads model and weights
        self.base_model = create_base_model()
        load_weights(self.base_model,
                     self.base_model_paths[fold],
                     cpu_only=self.cpu_only)
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

        def use_pos_batches(x: torch.Tensor, batches_mask: torch.Tensor):
            pos_idx = [
                batch_idx for batch_idx, boolean in enumerate(batches_mask)
                if boolean
            ]
            return x[pos_idx]

        # Single GPU inference
        for batch in self.test_loader:
            pred = predict_binary(filter_model, batch[0])
            # indices of the batch that the model thought were GWs
            pos_idx = pred >= self.threshold
            if pos_idx.any():
                pos_batch = use_pos_batches(batch[0], pos_idx)
                pred[pos_idx] = predict_binary(base_model, pos_batch)
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
        both_metrics = self.infer_both(self.filter_model, self.base_model)
        fold_metrics = {**mr_only_metrics, **base_only_metrics, **both_metrics}
        return fold_metrics

    def infer_all(self):
        """Inference on all folds
        """
        for fold in range(self.num_folds):
            print(f"Running inference for fold {fold}")
            metrics = self.infer_single_fold(fold)
            fold_metrics = {"fold": fold, **metrics}
            self.metrics = self.metrics.append(fold_metrics, ignore_index=True)

        avg = {}
        for metric in self.cols:
            if metric == "fold":
                avg["fold"] = "mean"
            else:
                # metric_avg = self.metrics[metric].mean()
                # metric_std = self.metrics[metric].std()
                # avg[metric] = f"{metric_avg} +/- {metric_std}"
                avg[metric] = self.metrics[metric].mean()

        stddev = {}
        for metric in self.cols:
            if metric == "fold":
                stddev["fold"] = "stddev"
            else:
                stddev[metric] = self.metrics[metric].std()

        self.metrics = self.metrics.append(avg, ignore_index=True)
        self.metrics = self.metrics.append(stddev, ignore_index=True)
        print("Results: \n", self.metrics)


def infer_auc(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Wraps sklearn.metrics.roc_auc_score to work with torch tensors.
    """
    y = y_true.detach().cpu().numpy().flatten()
    pred = y_pred.detach().cpu().numpy().flatten()
    return roc_auc_score(y, pred)


def predict_binary(model, x: torch.Tensor) -> torch.Tensor:
    """Assumes that the model returns logits. The output activation function is
    sigmoid.

    Args:
        model: The model to predict with
        x: the input batch
    """
    sigmoid = torch.nn.Sigmoid()
    return sigmoid(model(x))


def load_weights(model: torch.nn.Module,
                 weights_path: str,
                 cpu_only: bool = False):
    """Loads weights into model from weights_path.

    The state dict from the baseline has the keys:
        "global_epoch", "model", "state", "all_states"
    To load the model weights, only use state_dict["model"].

    The minirocket version has keys:
        dict_keys(['run_key', 'global_epoch_step', 'global_batch_step',
        'global_sample_step', 'stage_key', 'stage_epoch_step',
        'stage_batch_step', 'stage_sample_step', 'epoch_metrics',
        'loader_key', 'loader_batch_step', 'loader_sample_step', 
        'checkpointer_loader_key', 'checkpointer_metric_key',
        'checkpointer_minimize', 'model_state_dict'])

    To load the model weights, only use state_dict["model_state_dict"]
    """
    if cpu_only:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    else:
        state_dict = torch.load(weights_path)

    # base model
    if "model" in state_dict.keys():
        key = "model"
    # catalyst
    elif "model_state_dict" in state_dict.keys():
        key = "model_state_dict"

    model.load_state_dict(state_dict[key])


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