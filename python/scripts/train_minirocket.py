from fastai.basics import Learner, DataLoaders
from fastai.text.all import ShowGraphCallback
from fastai.metrics import RocAucBinary
from fastai.torch_core import default_device

from pathlib import Path
import numpy as np
from g2net.models.filter import MiniRocketFeatures, MiniRocketHead, \
                                MiniRocket, get_minirocket_features
from g2net.utils.tsai import Timer
from g2net.utils import build_ts_model
import torch


def pipeline(X: np.array,
             train_loader: torch.data.DataLoader,
             valid_loader: torch.data.DataLoader,
             learn_feats_batchwise: bool = False):
    dls = DataLoaders(train_loader, valid_loader)
    dls.vars = 3  # channels in, 3 signals
    dls.c = 1  # channels out (1 binary classification)
    dls.len = 4096  # length of a single signal

    if learn_feats_batchwise:
        # Online mode; head is learned
        model = build_ts_model(MiniRocket, dls=dls)
    else:
        # Offline mode
        # Fit minirocket features on a batch
        mrf = MiniRocketFeatures(X.shape[1], X.shape[2]).to(default_device())
        mrf.fit(X)
        X_feat = get_minirocket_features(X, mrf, chunksize=1024, to_np=True)
        # model is a linear classifier Head
        model = build_ts_model(MiniRocketHead, dls=dls)

    # Drop into fastai and use it to find a good learning rate.
    learn = Learner(dls, model, metrics=RocAucBinary, cbs=ShowGraphCallback())
    learn.lr_find()

    learn = Learner(dls, model, metrics=RocAucBinary, cbs=ShowGraphCallback())
    timer = Timer()
    timer.start()
    learn.fit_one_cycle(10, 3e-4)
    timer.stop()

    PATH = Path('./models/MiniRocket_aug.pkl')

    PATH.parent.mkdir(parents=True, exist_ok=True)
    learn.export(PATH)