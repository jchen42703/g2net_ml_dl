from pathlib import Path
import pandas as pd
from os.path import isfile
from g2net.logger import TorchLogger


def create_train_and_test_sub_csvs(config: dict):
    """Creates the paths for the training and sample submission test files.
    Note: For our experiments, we'll just use the training files and split them
    for the final evaluation (lets us sample unbalanced distributions easier)

    Assumes that the directory structure is

    data_dir
        train/
        test/ (optional)
        training_labels.csv
        sample_submission.csv (optional)

    Args:
        config: yml config w/ data_dir and export_dir
    Returns:
        None
    """
    LOGGER = TorchLogger('tmp.log', file=False)

    data_dir = config.get("data_dir")
    if data_dir == None:
        raise ValueError("data_dir must be specified in the config")
    root_dir = Path(data_dir).expanduser()
    train = pd.read_csv(root_dir / 'training_labels.csv')

    export_dir = config.get("export_dir")
    if export_dir == None:
        raise ValueError("export_dir must be specified in the config")
    export_dir = Path(export_dir).expanduser()
    export_dir.mkdir(parents=True, exist_ok=True)

    LOGGER('===== PROCESSING TRAINING CSV =====')
    train['path'] = train['id'].apply(
        lambda x: root_dir / f'train/{x[0]}/{x[1]}/{x[2]}/{x}.npy')
    train.to_csv(export_dir / 'train.csv', index=False)

    LOGGER('===== PROCESSING SAMPLE SUBMISSION CSV =====')
    test_csv = root_dir / 'sample_submission.csv'
    if isfile(test_csv):
        test = pd.read_csv(test_csv)
        test['path'] = test['id'].apply(
            lambda x: root_dir / f'test/{x[0]}/{x[1]}/{x[2]}/{x}.npy')
        test.to_csv(export_dir / 'test.csv', index=False)
    else:
        LOGGER('===== No sample_submission.csv found. =====')
