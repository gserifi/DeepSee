import lightning as lit
import torch


class BaseFeatureExtractor(lit.LightningModule):
    """
    Base class for feature extractors
    """

    def __init__(self):
        super().__init__()
        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "The forward method must be implemented in the subclass"
        )
