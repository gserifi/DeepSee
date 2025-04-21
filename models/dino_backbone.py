import torch
import torch.nn.functional as F
from torch import nn, optim
from transformers import AutoImageProcessor, Dinov2WithRegistersBackbone

from base_model import LitBaseModel


class DinoBackbone(LitBaseModel):
    def __init__(self, dino_model: str = "facebook/dinov2-with-registers-base"):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(
            dino_model, use_fast=True
        )

        self.out_features = ["stage2", "stage5", "stage8", "stage11"]
        self.dino = Dinov2WithRegistersBackbone.from_pretrained(
            dino_model, out_features=self.out_features
        )

        self.patch_size = self.dino.config.patch_size
        self.hidden_size = int(self.dino.config.hidden_size)
        self.num_features = len(self.out_features)

        self.feat_channels = self.num_features * self.hidden_size

        # For images of shape (B, 3, 224, 224) features have shape (B, N * 768, 16, 16)

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

    def forward(self, x):
        inputs = self.image_processor(images=x, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        outputs = self.dino(**inputs)
        feat_maps = torch.cat(outputs.feature_maps, dim=1)

        x1 = F.interpolate(
            x, size=feat_maps.shape[2:], mode="bicubic", align_corners=False
        )
        d1 = self.bn1(self.conv1(torch.cat([x1, feat_maps], dim=1)))
        d1 = self.up1(d1)
        d1 = F.gelu(d1)

        x2 = F.interpolate(x, size=d1.shape[2:], mode="bicubic", align_corners=False)
        d2 = self.bn2(self.conv2(torch.cat([x2, d1], dim=1)))
        d2 = self.up2(d2)
        d2 = F.gelu(d2)

        x3 = F.interpolate(x, size=d2.shape[2:], mode="bicubic", align_corners=False)
        d3 = self.conv3(torch.cat([x3, d2], dim=1))

        depth = F.interpolate(
            d3,
            size=x.shape[2:],
            mode="bicubic",
            align_corners=False,
        )

        depth = torch.sigmoid(depth) * 10

        return depth

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters())
        return optimizer
