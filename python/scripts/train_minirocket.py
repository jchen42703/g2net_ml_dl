from g2net.train import TrainPipeline, create_base_transforms, create_dataloaders
from g2net.train import create_base_transforms, create_dataloaders
from datetime import datetime
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from typing import List
import pprint


def train_one_fold(fold: int,
                   seed: int,
                   train_loader: DataLoader,
                   valid_loader: DataLoader,
                   pipeline_params: dict = None):
    """Trains one fold

    Args:
        pipeline_params: Corresponds to the pipeline params config.
            See train_minirocket.yml
    """
    timestamp = datetime.now().timestamp()
    params = {
        "train_loader": train_loader,
        "valid_loader": valid_loader,
    }

    pipeline_params["model_params"]["random_state"] = seed
    model_path = f"minirocket_rocket_fold{fold}_seed{seed}_{timestamp}.pt"
    params = {**params, **pipeline_params, "save_path": model_path}
    pp = pprint.PrettyPrinter(depth=4)
    print("PIPELINE PARAMS:")
    pp.pprint(params)
    pipeline = TrainPipeline(**params)
    pipeline.train_minirocket()


def prep_CV(train: pd.DataFrame,
            seed: int,
            num_splits: int = 5) -> List[List[List[int]]]:
    """Loads the components need for KFolds
    """
    splitter = StratifiedKFold(n_splits=num_splits,
                               shuffle=True,
                               random_state=seed)
    fold_iter = list(splitter.split(X=train, y=train['target']))
    return fold_iter


def create_fold_dl(train: pd.DataFrame,
                   train_idx: List[int],
                   valid_idx: List[int],
                   batch_size: int = 64):
    """Creates the fold subset dfs and dataloaders.
    
    Args:
        fold_iter: from kfolds
    """
    train_fold = train.iloc[train_idx]
    valid_fold = train.iloc[valid_idx]
    print(
        f'train positive: {train_fold.target.values.mean(0)} ({len(train_fold)})'
    )
    print(
        f'valid positive: {valid_fold.target.values.mean(0)} ({len(valid_fold)})'
    )
    transforms = create_base_transforms()
    train_loader, valid_loader = create_dataloaders(
        train_fold,
        valid_fold,
        batch_size=batch_size,
        train_transforms=transforms["train"],
        test_transforms=transforms["test"])
    return (train_loader, valid_loader)


if __name__ == "__main__":
    import argparse
    import os
    from g2net.utils.config_reader import load_config
    from g2net.utils.torch import seed_everything

    parser = argparse.ArgumentParser(description="For training.")
    parser.add_argument("--yml_path",
                        type=str,
                        required=True,
                        help="Path to the .yml config.")
    args = parser.parse_args()

    cfg = load_config(args.yml_path)["train_config"]

    print("CONFIG: \n", cfg)

    # Cross validation
    seed = cfg["seed"]
    num_splits = cfg["num_splits"]
    seed_everything(cfg["seed"], deterministic=False)
    train_path = os.path.join(cfg["dset_dir"], "train.csv")
    train = pd.read_csv(train_path).iloc[:cfg["dset_size"]]
    print(f"Creating {num_splits} folds with seed {seed}...")
    fold_iter = prep_CV(train, seed, num_splits=num_splits)

    # Training for cfg.num_splits folds
    orig_logdir = cfg["pipeline_params"]["logdir"]
    for fold, (train_idx, valid_idx) in enumerate(fold_iter):
        print(f"======== TRAINING FOLD {fold} ========")
        train_loader, valid_loader = create_fold_dl(
            train, train_idx, valid_idx, batch_size=cfg["batch_size"])
        cfg["pipeline_params"]["logdir"] = os.path.join(orig_logdir,
                                                        f"logs_{fold}")
        train_one_fold(fold,
                       seed,
                       train_loader=train_loader,
                       valid_loader=valid_loader,
                       pipeline_params=cfg["pipeline_params"])
