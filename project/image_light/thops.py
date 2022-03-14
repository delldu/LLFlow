import torch
from typing import List


def sum(tensor, dim: List[int]):
    dim = sorted(dim)
    for d in dim:
        tensor = tensor.sum(dim=d, keepdim=True)
    for i, d in enumerate(dim):
        tensor.squeeze_(d - i)
    return tensor


def mean(tensor, dim: List[int]):
    dim = sorted(dim)
    for d in dim:
        tensor = tensor.mean(dim=d, keepdim=True)
    return tensor


def split_cross(tensor):
    return tensor[:, 0::2, ...], tensor[:, 1::2, ...]


def cat_feature(tensor_a, tensor_b):
    return torch.cat((tensor_a, tensor_b), dim=1)


def pixels(tensor):
    return int(tensor.size(2) * tensor.size(3))
