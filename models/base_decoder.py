from typing import Sequence

import lightning as lit
import torch


class BaseDecoder(lit.LightningModule):
    """
    Base class for feature decoders
    """

    def __init__(self, feature_shape: Sequence[int]):
        """
        :param feature_shape: Shape of the feature maps to be decoded
        """
        super().__init__()
        self.save_hyperparameters()
        self.feature_shape = feature_shape

    def forward(
        self, x: torch.Tensor, feats: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: Input images of shape (B, 3, H, W)
        :param feats: Tuple of feature maps of shape (B, FD, PH, PW)
        :return: Depth map and log variance of shape (B, 1, H, W)
        """

        raise NotImplementedError(
            "The forward method must be implemented in the subclass"
        )
