from typing import Sequence

import torch
import torch.nn.functional as F

from models.base_decoder import BaseDecoder


class ResidualConvBlock(torch.nn.Module):
    """
    Residual convolutional block with two convolutional layers and batch normalization.
    Conserves input size and uses ReLU activation.

    As from:
    - RefineNet: Multi-Path Refinement Networks for High-Resolution Semantic Segmentation (https://arxiv.org/abs/1611.06612)
    And used in:
    - Vision Transformers for Dense Prediction (https://arxiv.org/abs/2103.13413)
    - Depth-Anything-V2 (https://arxiv.org/abs/2406.09414)

    Adapted from https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/util/blocks.py
    """

    def __init__(self, in_channels: int, out_channels: int):
        super(ResidualConvBlock, self).__init__()
        self.conv_block1 = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
            ),
            torch.nn.BatchNorm2d(out_channels),
        )
        self.conv_block2 = torch.nn.Sequential(
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=1,
            ),
            torch.nn.BatchNorm2d(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x1 = self.conv_block1(x)
        x2 = self.conv_block2(x1)
        return residual + x2


class FusionBlock(torch.nn.Module):
    """
    Fusion block to continuously fuse features from different stages of the encoder.
    As from:
    - Vision Transformers for Dense Prediction (https://arxiv.org/abs/2103.13413)
    And used in:
    - Depth-Anything-V2 (https://arxiv.org/abs/2406.09414)

    Adapted from https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/util/blocks.py
    """

    def __init__(self, in_channels: int):
        super().__init__()
        self.rcu1 = ResidualConvBlock(in_channels, in_channels)
        self.rcu2 = ResidualConvBlock(in_channels, in_channels)
        self.project = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None) -> torch.Tensor:
        """
        :param x: reassembled feature tensor (B, C, H, W)
        :param residual: residual connection from the previous fusion block (B, C, H, W)
        """
        if residual is not None:
            x = self.rcu1(x)
            x = x + residual
        x = self.rcu2(x)

        # Resample to 2x the original size (In DPT paper: Resample_{0.5})
        # (B, C, H, W) -> (B, C, 2H, 2W)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

        # Project
        return self.project(x)  # 1x1 convolution to reduce channels


class ReassembleBlock(torch.nn.Module):
    """
    Reassemble block to transform image tokens into image-like feature representations.
    As from:
    - Vision Transformers for Dense Prediction (https://arxiv.org/abs/2103.13413)

    Adapted from https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/dpt.py
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        features: int,
        n_patches_h: int,
        n_patches_w: int,
        scale: float = 1.0,
    ):
        """
        :param in_channels: Number of input channels (feature channels from DinoV2)
        :param out_channels: Number of output channels (feature channels for DPT)
        :param features: Additional output features for fusion (only in Depth-Anything-V2)
        :param n_patches_h: Number of patches in height
        :param n_patches_w: Number of patches in width
        :param scale: Scale factor for up-sampling (4.0, 2.0, 1.0, 0.5) (only in Depth-Anything-V2)

        Note that compared to original DPT, Depth-Anything-V2 uses an additional projection layer.
        """
        super().__init__()
        self.n_patches_h = n_patches_h
        self.n_patches_w = n_patches_w
        self.proj = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, stride=1, padding=0
        )

        # Use Depth-Anything-V2 resampling method instead of DPT
        if scale == 4.0:
            self.resample = torch.nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=4, stride=4, padding=0
            )
        elif scale == 2.0:
            self.resample = torch.nn.ConvTranspose2d(
                out_channels, out_channels, kernel_size=2, stride=2, padding=0
            )
        elif scale == 1.0:
            self.resample = torch.nn.Identity()
        elif scale == 0.5:
            self.resample = torch.nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            raise ValueError(f"Unknown scale value. {scale}")

        # Add the fusion projection layer from Depth-Anything-V2.
        # This is missing in the original DPT code.
        self.fusion_proj = torch.nn.Conv2d(
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
        assert (
            N == self.n_patches_h * self.n_patches_w
        ), f"Expected {self.n_patches_h * self.n_patches_w} tokens, got {N}"
        x = (
            x.permute(0, 2, 1)
            .reshape(B, C, self.n_patches_h, self.n_patches_w)
            .contiguous()
        )
        x = self.proj(x)
        x = self.resample(x)
        return self.fusion_proj(x)  # Additional Depth-Anything-V2 projection layer


class DepthHead(torch.nn.Module):
    """
    Depth head to predict the depth map from the fused features.
    Adapted from https://github.com/DepthAnything/Depth-Anything-V2/blob/main/depth_anything_v2/util/dpt.py
    """

    def __init__(
        self, in_channels: int, patch_h: int, patch_w: int, max_depth: int = 10
    ):
        """
        :param in_channels: Number of input channels (feature channels from DPT)
        :param patch_h: Number of patches in height
        :param patch_w: Number of patches in width
        :param max_depth: Maximum depth value (10 meters as per project description)
        """
        super().__init__()
        self.n_patch_h = patch_h
        self.n_patch_w = patch_w
        self.max_depth = max_depth
        self.conv1 = torch.nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1
        )
        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels // 2, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, output_dim: Sequence[int]) -> torch.Tensor:
        x = self.conv1(x)
        x = F.interpolate(
            x,
            size=(self.n_patch_h * 14, self.n_patch_w * 14),
            mode="bilinear",
            align_corners=True,
        )
        depth = self.conv_block_2(x)

        depth = depth * self.max_depth

        depth = F.interpolate(
            depth,
            size=output_dim,
            mode="bilinear",
            align_corners=True,
        )
        return depth


class LogVarHead(torch.nn.Module):
    """
    Log variance head to predict the log variance of the depth map.
    Currently same as depth head, but without sigmoid activations.
    """

    def __init__(self, in_channels: int, patch_h: int, patch_w: int):
        """
        :param in_channels: Number of input channels (feature channels from DPT)
        :param patch_h: Number of patches in height
        :param patch_w: Number of patches in width
        """
        super().__init__()
        self.n_patches_h = patch_h
        self.n_patches_w = patch_w
        self.conv1 = torch.nn.Conv2d(
            in_channels, in_channels // 2, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels // 2, 1, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, output_dim: Sequence[int]) -> torch.Tensor:
        x = self.conv1(x)
        x = F.interpolate(
            x,
            size=(self.n_patches_h * 14, self.n_patches_w * 14),
            mode="bilinear",
            align_corners=True,
        )
        logvar = self.conv2(x)

        logvar = F.interpolate(
            logvar,
            size=output_dim,
            mode="bilinear",
            align_corners=True,
        )
        return logvar


class PDPTDecoder(BaseDecoder):
    """
    Probabilistic DPT decoder, based on:
    - Vision Transformers for Dense Prediction (https://arxiv.org/abs/2103.13413)
    - Depth-Anything-V2 (https://arxiv.org/abs/2406.09414)

    PDPT decoder adds a probabilistic depth head to the DPT decoder,
    which can be used for uncertainty estimation in depth prediction.
    """

    def __init__(
        self,
        feature_shape: Sequence[int],
        feature_dim: int = 256,
        out_channels: Sequence[int] = (256, 512, 1024, 1024),
        debug: bool = False,
    ):
        """
        :param feature_shape: (feature channels, patches in height, patches in width)
        :param feature_dim: Dimension of feature space used in decoder.
                            Defaults to 256 (Depthy-Anything-V2).
        :param out_channels: Number of channels in each reassemble block.
        :param debug: If true, print debug information.
        """
        super().__init__(feature_shape)
        self.debug = debug

        if self.debug:
            print(f"Feature shape: {feature_shape}")

        self.feat_channels: int = self.feature_shape[0]
        self.n_patches_h: int = self.feature_shape[1]
        self.n_patches_w: int = self.feature_shape[2]
        self.feature_dim = feature_dim

        # Reassemble blocks: Transform image tokens into image-like feature
        # representations, which are then continuously fused for the final depth estimation.
        self.reassemble_block1 = ReassembleBlock(
            in_channels=self.feat_channels,
            out_channels=out_channels[0],
            features=feature_dim,
            n_patches_h=self.n_patches_h,
            n_patches_w=self.n_patches_w,
            scale=4.0,  # 4x spatial up-sampling (Depth-Anything-V2)
        )

        self.reassemble_block2 = ReassembleBlock(
            in_channels=self.feat_channels,
            out_channels=out_channels[1],
            features=feature_dim,
            n_patches_h=self.n_patches_h,
            n_patches_w=self.n_patches_w,
            scale=2.0,  # 2x spatial up-sampling (Depth-Anything-V2)
        )

        self.reassemble_block3 = ReassembleBlock(
            in_channels=self.feat_channels,
            out_channels=out_channels[2],
            features=feature_dim,
            n_patches_h=self.n_patches_h,
            n_patches_w=self.n_patches_w,
            scale=1.0,  # 1x spatial up-sampling (Depth-Anything-V2)
        )

        self.reassemble_block4 = ReassembleBlock(
            in_channels=self.feat_channels,
            out_channels=out_channels[3],
            features=feature_dim,
            n_patches_h=self.n_patches_h,
            n_patches_w=self.n_patches_w,
            scale=0.5,  # 0.5x spatial up-sampling (Depth-Anything-V2)
        )

        # Fusion blocks: Continuously fuse the reassembled features to create a final depth map.
        self.fusion_block1 = FusionBlock(in_channels=feature_dim)
        self.fusion_block2 = FusionBlock(in_channels=feature_dim)
        self.fusion_block3 = FusionBlock(in_channels=feature_dim)
        self.fusion_block4 = FusionBlock(in_channels=feature_dim)

        # Depth head: Predicts the depth map from the fused features (Depth-Anything-V2).
        self.depth_head = DepthHead(
            in_channels=feature_dim,
            patch_h=self.n_patches_h,
            patch_w=self.n_patches_w,
        )

        # Log variance head: Predicts the log variance of the depth map (ours).
        self.logvar_head = LogVarHead(
            in_channels=feature_dim,
            patch_h=self.n_patches_h,
            patch_w=self.n_patches_w,
        )

    def forward(
        self, x: torch.Tensor, feats: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param x: Input image tensor (B, C, H, W)
        :param feats: Tuple of feature tensors from the encoder (B, num_patches, feature_dim).
                      Each tensor corresponds to a different stage of the encoder.
        """

        # Reassemble the features
        # (B, num_patches, dpt_feature_dim) -> (B, DAV2_feautre_dim, patch_h * scale, patch_w * scale)
        x1 = self.reassemble_block1(feats[0])  # Scale 4.0
        x2 = self.reassemble_block2(feats[1])  # Scale 2.0
        x3 = self.reassemble_block3(feats[2])  # Scale 1.0
        x4 = self.reassemble_block4(feats[3])  # Scale 0.5

        if self.debug:
            print(f"Reassemble block 4: {feats[3].shape} -> {x4.shape}")
            print(f"Reassemble block 3: {feats[2].shape} -> {x3.shape}")
            print(f"Reassemble block 2: {feats[1].shape} -> {x2.shape}")
            print(f"Reassemble block 1: {feats[0].shape} -> {x1.shape}")

        # Apply fusion blocks (Up-scale 2x) to continuously fuse the features
        x4 = self.fusion_block1(x4)  # No Residual connection
        x3 = self.fusion_block2(x3, residual=x4)
        x2 = self.fusion_block3(x2, residual=x3)
        x1 = self.fusion_block4(x1, residual=x2)

        if self.debug:
            print(f"Fusion block 4: {x4.shape}")
            print(f"Fusion block 3: {x3.shape}")
            print(f"Fusion block 2: {x2.shape}")
            print(f"Fusion block 1: {x1.shape}")

        # Final output, shape is passed for bilinear interpolation to original size
        depth = self.depth_head(x1, x.shape[2:])
        logvar = self.logvar_head(x1, x.shape[2:])

        return depth, logvar
