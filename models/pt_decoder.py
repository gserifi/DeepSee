from typing import Literal, Sequence

import einops
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor

from models.base_decoder import BaseDecoder


class PTransformerDecoder(BaseDecoder):
    def __init__(
        self,
        feature_shape: Sequence[int],
        decoder_dim: int = 2,
        nhead: int = 4,
        num_layers: int = 3,
        tgt_mode: Literal["learned", "image"] = "learned",
    ):
        super().__init__(feature_shape)

        self.decoder_dim = decoder_dim
        self.tgt_mode = tgt_mode

        self.patch_size = 14

        self.PH, self.PW = self.feature_shape[1], self.feature_shape[2]
        self.ph, self.pw = self.patch_size, self.patch_size
        self.C = self.decoder_dim

        d_model = self.C * self.ph * self.pw

        decoder_layer = torch.nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
        )

        self.decoder = torch.nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
        )

        self.feat_proj = torch.nn.Conv1d(self.feature_shape[0], d_model, kernel_size=1)

        if self.tgt_mode == "image":
            self.image_processor = AutoImageProcessor.from_pretrained(
                "facebook/dinov2-with-registers-base",
                use_fast=True,
                crop_size={"height": self.PH * self.ph, "width": self.PW * self.pw},
                size={"shortest_edge": self.PH * self.ph},
            )

            self.image_proj = torch.nn.Conv1d(
                3 * self.ph * self.pw, d_model, kernel_size=1
            )
        elif self.tgt_mode == "learned":
            self.tgt_tokens = torch.nn.Parameter(
                torch.randn(self.PH * self.PW, d_model)
            )
        else:
            raise ValueError(
                f"Invalid tgt_mode: {self.tgt_mode}. Must be 'learned' or 'image'."
            )

        self.head = torch.nn.Conv2d(self.decoder_dim, 2, kernel_size=1)

    def forward(
        self, x: torch.Tensor, feats: tuple[torch.Tensor, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        feats = torch.cat(feats, dim=-1)

        feats = self.feat_proj(feats.permute(0, 2, 1)).permute(0, 2, 1)

        tgt = None
        if self.tgt_mode == "image":
            tgt = self.image_processor(images=x, return_tensors="pt", do_rescale=False)
            tgt = tgt["pixel_values"].to(x.device)
            tgt = einops.rearrange(
                tgt,
                "b c (h ph) (w pw) -> b (h w) (c ph pw)",
                h=self.PH,
                w=self.PW,
                ph=self.ph,
                pw=self.pw,
                c=x.size(1),
            )
            tgt = self.image_proj(tgt.permute(0, 2, 1)).permute(0, 2, 1)
        elif self.tgt_mode == "learned":
            tgt = self.tgt_tokens.unsqueeze(0).expand(B, -1, -1)

        out = self.decoder(tgt, feats)

        out = einops.rearrange(
            out,
            "b (h w) (c ph pw) -> b c (h ph) (w pw)",
            h=self.PH,
            w=self.PW,
            ph=self.ph,
            pw=self.pw,
            c=self.C,
        )

        out = self.head(out)

        depth = out[:, 0:1, :, :]
        logvar = out[:, 1:2, :, :]

        depth = F.interpolate(depth, size=(x.size(2), x.size(3)), mode="nearest")
        logvar = F.interpolate(logvar, size=(x.size(2), x.size(3)), mode="nearest")

        depth = torch.sigmoid(depth) * 10

        return depth, logvar
