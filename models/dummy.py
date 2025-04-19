import torch
from torch import nn, optim

from base_model import LitBaseModel


class Dummy(LitBaseModel):
    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.randn(1, 426, 560, requires_grad=True))

    def forward(self, x):
        return x[:, 0:1, :, :] + self.offset.unsqueeze(0).repeat(x.size(0), 1, 1, 1)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
