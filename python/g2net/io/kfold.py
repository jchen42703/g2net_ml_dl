import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from typing import List


def split_into_train_val_test(train_df: pd.DataFrame, seed: int = 420):
    """Conducts a single train/val/test split of the dataset for the provided
    seed.

    Returns:
        train_idx ()
    """
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    train_idx, test_valid_idx = next(split.split(train_df, train_df.target))

    # Now split the test_valid_idx into test_idx and valid_idx separately
    test_valid_set = train_df.iloc[test_valid_idx]
    split2 = StratifiedShuffleSplit(n_splits=1,
                                    test_size=0.5,
                                    random_state=seed)
    test_idx, valid_idx = next(
        split2.split(test_valid_set, test_valid_set.target))
    return train_idx, valid_idx, test_idx


def getKFolds(train_df: pd.DataFrame, seeds: List[str]) -> List[List[int]]:
    """Generates len(seeds) folds for train_df

    Usage:
        # 5 folds
        folds = getKFolds(train_df, [42, 99, 420, 120, 222])
        for fold, (train_idx, valid_idx) in enumerate(folds):
            train_fold = train.iloc[train_idx]
            valid_fold = train.iloc[valid_idx]
            ...

    Returns:
        folds: list of [train, val, test] indices for each 
    """
    folds = []
    for seed in seeds:
        train, val, test = split_into_train_val_test(train_df, seed=seed)
        folds.append([train, val, test])
    return folds
