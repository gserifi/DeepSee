import torch
from torch import optim

from base_model import LitBaseModel
from models.base_decoder import BaseDecoder
from models.base_feature_extractor import BaseFeatureExtractor


class FeatureDecoder(LitBaseModel):
    """
    Feature extractor + Decoder model.
    """

    def __init__(
        self,
        feature_extractor: BaseFeatureExtractor,
        decoder: BaseDecoder,
        freeze_extractor: bool = True,
    ):
        """
        :param feature_extractor: Feature extractor model
        :param decoder: Decoder model
        :param freeze_extractor: If True, freeze the feature extractor parameters
        """

        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder
        self.freeze_extractor = freeze_extractor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model. The input is a batch of images and the output is a depth map of the same shape.
        :param x: Input images of shape (B, 3, H, W)
        :return: Depth map of shape (B, 1, H, W)
        """

        feats = self.feature_extractor(x)
        depth = self.decoder(x, feats)

        return depth

    def configure_optimizers(self) -> optim.Optimizer:
        # if self.freeze_extractor:
        #     for param in self.feature_extractor.parameters():
        #         param.requires_grad = False
        #
        #     optimizer = optim.Adam(self.decoder.parameters(), lr=1000)
        # else:
        #     optimizer = optim.Adam(self.parameters(), lr=1000)

        lr = 0.000005
        optimizer = optim.AdamW(
            [
                {
                    "params": self.decoder.parameters(),
                    "lr": lr * 10.0,
                },
            ],
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )

        return optimizer
