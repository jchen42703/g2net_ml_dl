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


def create_subset_df(train_dir: str, df_path: str):
    """Creates a dataframe with only the ids of the files located in the
    train_dir.

    Args:
        train_dir: Directory to where all of the files are located
        df_path: Path to the dataframe to subset (generally should be the
            train.csv)
    """
    labels = pd.read_csv(df_path)
    all_paths = get_all_paths(train_dir=train_dir)
    return get_subset_df(all_paths, labels)
