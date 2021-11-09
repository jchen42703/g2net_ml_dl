import os
from glob import glob
from typing import List
import pandas as pd
from pathlib import Path


def get_all_paths(train_dir: str) -> List[str]:
    return glob(os.path.join(train_dir, "*/*/*/*.npy"))


def get_subset_df(paths: List[str], labels: pd.DataFrame) -> List[str]:
    # Gathers the ids that correspond to the file labels
    ids = [Path(p).stem for p in paths]
    df = labels.loc[labels["id"].isin(ids)]
    return df


def create_subset_df(train_dir: Path, labels_df_path: str):
    """Creates a dataframe with only the ids of the files located in the
    train_dir.
    """
    labels = pd.read_csv(labels_df_path)
    all_paths = get_all_paths(train_dir=train_dir)
    labels = get_subset_df(all_paths, labels)
    labels['path'] = labels['id'].apply(
        lambda x: train_dir / f'train/{x[0]}/{x[1]}/{x[2]}/{x}.npy')
    return labels
