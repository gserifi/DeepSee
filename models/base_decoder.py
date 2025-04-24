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

    def forward(self, x: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "The forward method must be implemented in the subclass"
        )
