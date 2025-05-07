from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_decoder import BaseDecoder


class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualConvBlock, self).__init__()
        self.conv_block1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_block2 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
            ),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        return residual + x2


class FusionBlock(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.rcu1 = ResidualConvBlock(in_channels, in_channels)
        self.rcu2 = ResidualConvBlock(in_channels, in_channels)
        self.project = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        if residual is not None:
            x = self.rcu1(x)
            x = x + residual
        x = self.rcu2(x)

        # Resample to 2x the original size
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Project
        return self.project(x)


class ReassembleBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: int,
        H: int,
        W: int,
        scale: float = 1.0,
    ):
        super().__init__()
        self.H = H
        self.W = W
        self.proj = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        # Use Depth-Anything-V2 resampling method instead of DPT
        if scale == 4.0:
            self.resample = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=4, stride=4, padding=0
            )
        elif scale == 2.0:
            self.resample = nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=2, stride=2, padding=0
            )
        elif scale == 1.0:
            self.resample = nn.Identity()
        elif scale == 0.5:
            self.resample = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            raise ValueError(f"Unknown scale value. {scale}")

        # Add the fusion projection layer from Depth-Anything-V2.
        # This is missing in the original DPT code.
        self.fusion_proj = nn.Conv2d(
            out_channels,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # We don't have a CLS token, so we just reshape input tensor
        B, N, C = x.shape
        assert N == self.H * self.W, f"Expected {self.H * self.W} tokens, got {N}"
        x = x.permute(0, 2, 1).reshape(B, C, self.H, self.W).contiguous()
        x = self.proj(x)
        x = self.resample(x)
        return self.fusion_proj(x)


class DepthHead(nn.Module):
    def __init__(
        self, in_channels: int, patch_h: int, patch_w: int, max_depth: int = 10
    ):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w
        self.max_depth = max_depth
        self.conv1 = nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, output_dim: Sequence[int]) -> torch.Tensor:
        x = self.conv1(x)
        print(f"path_h: {self.patch_h * 14}, patch_w: {self.patch_w * 14}")
        x = F.interpolate(
            x,
            size=(self.patch_h * 14, self.patch_w * 14),
            mode="bilinear",
            align_corners=True,
        )
        depth = self.conv_block_2(x)
        print(
            f"Min: {depth.min()}, Max: {depth.max()}, Mean: {depth.mean()}, Std: {depth.std()}"
        )

        depth = depth * self.max_depth

        depth = F.interpolate(
            depth,
            size=output_dim,
            mode="bilinear",
            align_corners=True,
        )
        return depth


class PDPTDecoder(BaseDecoder):
    def __init__(
        self,
        feature_shape: Sequence[int],
        feature_dim: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
    ):
        super().__init__(feature_shape)

        self.feat_channels: int = self.feature_shape[0]
        self.patch_h: int = self.feature_shape[1]
        self.patch_w: int = self.feature_shape[2]

        self.reassemble_block1 = ReassembleBlock(
            in_channels=self.feat_channels,
            out_channels=out_channels[0],
            features=feature_dim,
            H=self.patch_h,
            W=self.patch_w,
            scale=4.0,
        )

        self.reassemble_block2 = ReassembleBlock(
            in_channels=self.feat_channels,
            out_channels=out_channels[1],
            features=feature_dim,
            H=self.patch_h,
            W=self.patch_w,
            scale=2.0,
        )

        self.reassemble_block3 = ReassembleBlock(
            in_channels=self.feat_channels,
            out_channels=out_channels[2],
            features=feature_dim,
            H=self.patch_h,
            W=self.patch_w,
            scale=1.0,
        )

        self.reassemble_block4 = ReassembleBlock(
            in_channels=self.feat_channels,
            out_channels=out_channels[3],
            features=feature_dim,
            H=self.patch_h,
            W=self.patch_w,
            scale=0.5,
        )

        self.fusion_block1 = FusionBlock(feature_dim)
        self.fusion_block2 = FusionBlock(feature_dim)
        self.fusion_block3 = FusionBlock(feature_dim)
        self.fusion_block4 = FusionBlock(feature_dim)

        self.depth_head = DepthHead(
            in_channels=feature_dim,
            patch_h=self.patch_h,
            patch_w=self.patch_w,
        )

    def forward(self, x: torch.Tensor, feats: Tuple[torch.Tensor, ...]) -> torch.Tensor:
        # Reassemble the features
        x1 = self.reassemble_block1(feats[0])
        x2 = self.reassemble_block2(feats[1])
        x3 = self.reassemble_block3(feats[2])
        x4 = self.reassemble_block4(feats[3])

        print(f"Reassemble block 4: {feats[3].shape} -> {x4.shape}")
        print(f"Reassemble block 3: {feats[2].shape} -> {x3.shape}")
        print(f"Reassemble block 2: {feats[1].shape} -> {x2.shape}")
        print(f"Reassemble block 1: {feats[0].shape} -> {x1.shape}")

        # Apply fusion blocks
        x4 = self.fusion_block1(x4)
        print(f"Fusion block 4: {x1.shape}")
        x3 = self.fusion_block2(x3, x4)
        print(f"Fusion block 3: {x3.shape}")
        x2 = self.fusion_block3(x2, x3)
        print(f"Fusion block 2: {x2.shape}")
        x1 = self.fusion_block4(x1, x2)
        print(f"Fusion block 1: {x1.shape}")

        # Final output
        depth = self.depth_head(x1, x.shape[2:])
        print(f"Depth head output: {depth.shape}")

        return depth
