import pandas as pd
import numpy as np
from g2net.io.kfold import split_into_train_val_test, getKFolds


def test_split_into_train_val_test():
    # split_into_train_val_test must be deterministic for the same seed
    # (array([323374, 508502, 284104, ..., 208189, 114376, 107277]),
    # array([213467, 216819, 195261, ...,  96502,  42358, 167116]),
    # array([ 30501, 175959,  40720, ...,  74205, 163529, 205033]))
    train_df = pd.read_csv("./train.csv")
    train1, val1, test1 = split_into_train_val_test(train_df, seed=420)
    train2, val2, test2 = split_into_train_val_test(train_df, seed=420)
    assert np.array_equal(train1, train2)
    assert np.array_equal(val1, val2)
    assert np.array_equal(test1, test2)


def test_getKFolds():
    train_df = pd.read_csv("./train.csv")
    seeds = [1, 2, 3, 4, 5]
    # Tests that getKFolds is deterministic for the same list of seeds
    folds = getKFolds(train_df=train_df, seeds=seeds)
    folds2 = getKFolds(train_df=train_df, seeds=seeds)

    assert folds == folds2
