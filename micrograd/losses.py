import numpy as np

from .autograd import Tensor


def MSE(pred, target):
    diff = pred - target
    loss = (diff * diff).mean()
    return loss


def BinaryCrossEntropy(pred, target):
    pred = pred.clip(1e-7, 1.0 - 1e-7)
    loss = -(target * pred.log() + (1.0 - target) * (1.0 - pred).log()).mean()
    return loss


__all__ = ["MSE", "BinaryCrossEntropy"]
