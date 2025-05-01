from typing import Tuple

import lightning as lit
import torch


class BaseFeatureExtractor(lit.LightningModule):
    """
    Base class for feature extractors
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        :param x: Input images of shape (B, 3, H, W)

        :return: Tuple of feature maps of shape (B, FD, PH, PW)
        """
        raise NotImplementedError(
            "The forward method must be implemented in the subclass"
        )
