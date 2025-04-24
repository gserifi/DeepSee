import torch

from base_model import LitBaseModel
from models.base_decoder import BaseDecoder
from models.base_feature_extractor import BaseFeatureExtractor


class FeatureDecoder(LitBaseModel):
    """
    Feature extractor + Decoder model.
    """

    def __init__(self, feature_extractor: BaseFeatureExtractor, decoder: BaseDecoder):
        """
        :param feature_extractor: Feature extractor model
        :param decoder: Decoder model
        """

        super().__init__()
        self.feature_extractor = feature_extractor
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model. The input is a batch of images and the output is a depth map of the same shape.
        :param x: Input images of shape (B, 3, H, W)
        :return: Depth map of shape (B, 1, H, W)
        """

        feats = self.feature_extractor(x)
        depth = self.decoder(x, feats)

        return depth
