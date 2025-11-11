import torch
from torch._prims_common import DeviceLikeType


def get_device() -> DeviceLikeType:
    return "cuda" if torch.cuda.is_available() else "cpu"
