from typing import Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from models.base_decoder import BaseDecoder


class UNetDecoder(BaseDecoder):
    def __init__(self, feature_shape: Sequence[int]):
        super().__init__(feature_shape)

        self.feat_channels: int = self.feature_shape[0]

        # Define UNet decoder layers
        # For images of shape (B, 3, 224, 224), the output of the image processor, features are (B, N * 768, 16, 16)

        # conv1: (B, 3 + N * 768, 16, 16) -> (B, N * 192, 14, 14)
        # up1: (B, N * 192, 14, 14) -> (B, N * 192, 56, 56)
        self.conv1 = nn.Conv2d(
            3 + self.feat_channels,
            self.feat_channels // 4,
            kernel_size=3,
            stride=1,
            padding=0,
        )
        self.bn1 = nn.BatchNorm2d(self.feat_channels // 4)
        self.up1 = nn.Upsample(scale_factor=4, mode="bicubic", align_corners=False)

        # conv2: (B, 3 + N * 192, 56, 56) -> (B, N * 48, 56, 56)
        # up2: (B, N * 48, 56, 56) -> (B, N * 48, 224, 224)
        self.conv2 = nn.Conv2d(
            3 + self.feat_channels // 4,
            self.feat_channels // 16,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.bn2 = nn.BatchNorm2d(self.feat_channels // 16)
        self.up2 = nn.Upsample(scale_factor=4, mode="bicubic", align_corners=False)

        # conv3: (B, 3 + N * 48, 224, 224) -> (B, 1, 224, 224)
        self.conv3 = nn.Conv2d(
            3 + self.feat_channels // 16,
            1,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, x: torch.Tensor, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        feats = torch.cat(feats, dim=1)

        # At each step, downsample the input image to the size of the feature maps at that level and run the
        # concatenation through a convolutional layer, followed by upsampling and activation
        x1 = F.interpolate(x, size=feats.shape[2:], mode="bicubic", align_corners=False)
        d1 = self.bn1(self.conv1(torch.cat([x1, feats], dim=1)))
        d1 = self.up1(d1)
        d1 = F.gelu(d1)

        x2 = F.interpolate(x, size=d1.shape[2:], mode="bicubic", align_corners=False)
        d2 = self.bn2(self.conv2(torch.cat([x2, d1], dim=1)))
        d2 = self.up2(d2)
        d2 = F.gelu(d2)

        # Final convolutional layer that reduces the feature channels to a single depth channel
        x3 = F.interpolate(x, size=d2.shape[2:], mode="bicubic", align_corners=False)
        d3 = self.conv3(torch.cat([x3, d2], dim=1))

        # Upsample the depth map to the original image size
        depth = F.interpolate(
            d3,
            size=x.shape[2:],
            mode="bicubic",
            align_corners=False,
        )

        # Map the depth to a range of 0 to 10 meters, as stated in the project description
        depth = torch.sigmoid(depth) * 10

        return depth
