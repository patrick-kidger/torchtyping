import torch
import warnings


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    torch.rand(2, names=("a",))
