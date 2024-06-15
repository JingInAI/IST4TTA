import torch.nn as nn
import torch.nn.utils.weight_norm as WeightNorm

from .tools import init_weights


def linear(in_dim, out_dim, init=False, **kwargs):
    model = nn.Linear(in_dim, out_dim)
    if init:
        model.apply(init_weights)
    return model


def linear_wn(in_dim, out_dim, init=True, **kwargs):
    model = WeightNorm(nn.Linear(in_dim, out_dim), name="weight")
    if init:
        model.apply(init_weights)
    return model
