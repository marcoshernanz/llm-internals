import torch
from torch import nn


D_MODEL = 64


class RMSNorm(nn.Module):
    def __init__(self):
        super().__init__()

        weight = nn.Parameter(torch.ones(D_MODEL))

        self.eps = 1e-5

    def forward(self, x: torch.Tensor):
        square_mean = x.pow(2).mean(dim=-1)
        x * torch.rsqrt(square_mean + self.eps)
