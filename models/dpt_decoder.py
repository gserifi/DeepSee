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

        # # kaming normal
        # for m in self.depth_head.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         torch.nn.init.zeros_(m.weight)
        #         # torch.nn.init.kaiming_normal_(
        #         #     m.weight, mode="fan_out", nonlinearity="relu"
        #         # )
        #         if m.bias is not None:
        #             torch.nn.init.zeros_(m.bias)
        #     elif isinstance(m, torch.nn.BatchNorm2d):
        #         torch.nn.init.ones_(m.weight)
        #         torch.nn.init.zeros_(m.bias)
        #     elif isinstance(m, torch.nn.Linear):
        #         # torch.nn.init.kaiming_normal_(
        #         #     m.weight, mode="fan_out", nonlinearity="relu"
        #         # )
        #         torch.nn.init.zeros_(m.weight)
        #         if m.bias is not None:
        #             torch.nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        depth = self.depth_head(feats, self.patch_h, self.patch_w)

        print(
            f"Min: {depth.min()}, Max: {depth.max()}, Mean: {depth.mean()}, Std: {depth.std()}"
        )

        # depth = F.relu(1 - depth)

        # Map the depth to a range of 0 to 10 meters, as stated in the project description
        # depth = torch.sigmoid(depth) * 10
        depth = depth * 10
        # depth = torch.clamp(depth, 0, 10)

        depth = F.interpolate(
            depth,
            size=x.shape[2:],
            mode="bilinear",
            align_corners=True,
        )

        # print(f"Stats: {depth.min()}, {depth.max()}, {depth.mean()}, {depth.std()}")

        return depth
