from fastai.torch_core import default_device, apply_init
from fastai.layers import LinBnDrop, flatten_model
from fastai.basics import Learner
from g2net.utils.tsai import print_verbose
from functools import partial
from torch.nn import Flatten, Sequential, Linear, Module
from pathlib import Path
import torch


def is_linear(l):
    return isinstance(l, Linear)


def ifnone(a, b):
    # From fastai.fastcore
    "`b` if `a` is None else `a`"
    return b if a is None else a


class Reshape(Module):

    def __init__(self, *shape):
        self.shape = shape

    def forward(self, x):
        return x.reshape(x.shape[0], *self.shape)

    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(['bs'] + [str(s) for s in self.shape])})"


class create_lin_3d_head(Sequential):
    "Module to create a 3d output head with linear layers"

    def __init__(self,
                 n_in,
                 n_out,
                 seq_len,
                 d=(),
                 lin_first=False,
                 bn=True,
                 act=None,
                 fc_dropout=0.):

        assert len(
            d) == 2, "you must pass a tuple of len == 2 to create a 3d output"
        layers = [Flatten()]
        layers += LinBnDrop(n_in * seq_len,
                            n_out,
                            bn=bn,
                            p=fc_dropout,
                            act=act,
                            lin_first=lin_first)
        layers += [Reshape(*d)]

        super().__init__(*layers)


def noop(x):
    return x


def get_layers(model, cond=noop, full=True):
    if isinstance(model, Learner):
        model = model.model
    if full:
        return [m for m in flatten_model(model) if any([c(m) for c in L(cond)])]
    else:
        return [m for m in model if any([c(m) for c in L(cond)])]


def get_nf(m):
    "Get nf from model's first linear layer in head"
    return get_layers(m[-1], is_linear)[0].in_features


def transfer_weights(model,
                     weights_path: Path,
                     device: torch.device = None,
                     exclude_head: bool = True):
    """Utility function that allows to easily transfer weights between models.
    Taken from the great self-supervised repository created by Kerem Turgutlu.
    https://github.com/KeremTurgutlu/self_supervised/blob/d87ebd9b4961c7da0efd6073c42782bbc61aaa2e/self_supervised/utils.py"""

    device = ifnone(device, default_device())
    state_dict = model.state_dict()
    new_state_dict = torch.load(weights_path, map_location=device)
    matched_layers = 0
    unmatched_layers = []
    for name, param in state_dict.items():
        if exclude_head and 'head' in name:
            continue
        if name in new_state_dict:
            matched_layers += 1
            input_param = new_state_dict[name]
            if input_param.shape == param.shape:
                param.copy_(input_param)
            else:
                unmatched_layers.append(name)
        else:
            unmatched_layers.append(name)
            pass  # these are weights that weren't in the original model, such as a new head
    if matched_layers == 0:
        raise Exception("No shared weight names were found between the models")
    else:
        if len(unmatched_layers) > 0:
            print(f'check unmatched_layers: {unmatched_layers}')
        else:
            print(f"weights from {weights_path} successfully transferred!\n")


def build_ts_model(arch,
                   c_in=None,
                   c_out=None,
                   seq_len=None,
                   d=None,
                   dls=None,
                   device=None,
                   verbose=False,
                   pretrained=False,
                   weights_path=None,
                   exclude_head=True,
                   cut=-1,
                   init=None,
                   arch_config={},
                   **kwargs):

    device = ifnone(device, default_device())
    if dls is not None:
        c_in = ifnone(c_in, dls.vars)
        c_out = ifnone(c_out, dls.c)
        seq_len = ifnone(seq_len, dls.len)

    if sum([
            1 for v in [
                'RNN_FCN', 'LSTM_FCN', 'RNNPlus', 'LSTMPlus', 'GRUPlus',
                'InceptionTime', 'TSiT', 'GRU_FCN', 'OmniScaleCNN', 'mWDN',
                'TST', 'XCM', 'MLP', 'MiniRocket', 'InceptionRocket'
            ] if v in arch.__name__
    ]):
        print_verbose(
            f'arch: {arch.__name__}(c_in={c_in} c_out={c_out} seq_len={seq_len} device={device}, arch_config={arch_config}, kwargs={kwargs})',
            verbose)
        model = arch(c_in, c_out, seq_len=seq_len, **arch_config,
                     **kwargs).to(device=device)
    elif 'minirockethead' in arch.__name__.lower():
        print_verbose(
            f'arch: {arch.__name__}(c_in={c_in} seq_len={seq_len} device={device}, arch_config={arch_config}, kwargs={kwargs})',
            verbose)
        model = (arch(c_in, c_out, seq_len=1, **arch_config,
                      **kwargs)).to(device=device)
    elif 'rocket' in arch.__name__.lower():
        print_verbose(
            f'arch: {arch.__name__}(c_in={c_in} seq_len={seq_len} device={device}, arch_config={arch_config}, kwargs={kwargs})',
            verbose)
        model = (arch(c_in=c_in, seq_len=seq_len, **arch_config,
                      **kwargs)).to(device=device)
    else:
        print_verbose(
            f'arch: {arch.__name__}(c_in={c_in} c_out={c_out} device={device}, arch_config={arch_config}, kwargs={kwargs})',
            verbose)
        model = arch(c_in, c_out, **arch_config, **kwargs).to(device=device)

    try:
        model[0], model[1]
        subscriptable = True
    except:
        subscriptable = False
    if hasattr(model, "head_nf"):
        head_nf = model.head_nf
    else:
        try:
            head_nf = get_nf(model)
        except:
            head_nf = None

    if not subscriptable and 'Plus' in arch.__name__:
        model = Sequential(*model.children())
        model.backbone = model[:cut]
        model.head = model[cut:]

    if pretrained and not ('xresnet' in arch.__name__ and
                           not '1d' in arch.__name__):
        assert weights_path is not None, "you need to pass a valid weights_path to use a pre-trained model"
        transfer_weights(model,
                         weights_path,
                         exclude_head=exclude_head,
                         device=device)

    if init is not None:
        apply_init(model[1] if pretrained else model, init)

    setattr(model, "head_nf", head_nf)
    setattr(model, "__name__", arch.__name__)

    return model