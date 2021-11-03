import pandas as pd
from sklearn.model_selection import StratifiedKFold


def load_csv(csv_path, num_splits, seed, debug: bool = True):
    train_df = pd.read_csv(csv_path)
    if debug:
        train_df = train_df.iloc[:10000]
    splitter = StratifiedKFold(n_splits=num_splits,
                               shuffle=True,
                               random_state=seed)
    fold_iter = list(splitter.split(X=train_df, y=train_df['target']))
    # Usage
    # for fold, (train_idx, valid_idx) in enumerate(fold_iter):
    #     train_fold = train.iloc[train_idx]
    #     valid_fold = train.iloc[valid_idx]