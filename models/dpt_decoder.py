from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from ext.depth_anything_v2.dpt import DPTHead
from models.base_decoder import BaseDecoder


class DPTDecoder(BaseDecoder):
    def __init__(self, feature_shape: Sequence[int]):
        super().__init__(feature_shape)

        self.feat_channels: int = self.feature_shape[0]
        self.patch_h: int = self.feature_shape[1]
        self.patch_w: int = self.feature_shape[2]

        self.depth_head = DPTHead(self.feat_channels, use_bn=True)

    def forward(self, x: torch.Tensor, feats: Tuple[torch.Tensor]) -> torch.Tensor:
        depth = self.depth_head(feats, self.patch_h, self.patch_w)

        depth = F.interpolate(
            depth,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        # Map the depth to a range of 0 to 10 meters, as stated in the project description
        depth = torch.sigmoid(depth) * 10

        # print(f"Stats: {depth.min()}, {depth.max()}, {depth.mean()}, {depth.std()}")

        return depth
