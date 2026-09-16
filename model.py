import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()
