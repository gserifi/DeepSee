import torch
from transformers import AutoImageProcessor, Dinov2WithRegistersModel

from models.base_feature_extractor import BaseFeatureExtractor


class DinoFeatureExtractor(BaseFeatureExtractor):
    """
    Uses DINOv2 with registers (https://huggingface.co/docs/transformers/en/model_doc/dinov2_with_registers) as a
    feature extractor. The features are on a per-patch level.
    """

    def __init__(
        self,
        out_features: list[str] = None,
        dino_model: str = "facebook/dinov2-with-registers-base",
        feature_projection_dim: int = 128,
        use_feature_projection: bool = True,
    ):
        """
        :param dino_model: For other options, see https://huggingface.co/models?search=facebook/dino
        """

        super().__init__()

        self.feature_projection_dim = feature_projection_dim
        self.use_feature_projection = use_feature_projection

        # Load the image processor that transforms the input images into the format expected by the DINOv2 model
        self.image_processor = AutoImageProcessor.from_pretrained(
            dino_model,
            use_fast=True,
            crop_size={"height": 420, "width": 560},
            size={"shortest_edge": 420},
        )

        # Layer names to extract features from
        if out_features is None:
            out_features = ["stage2", "stage5", "stage8", "stage11"]
        self.out_features = out_features

        # Load the DINOv2 model with registers
        self.dino = Dinov2WithRegistersModel.from_pretrained(
            dino_model, out_features=out_features, reshape_hidden_states=True
        )

        self.patch_size: int = self.dino.config.patch_size
        self.hidden_size: int = self.dino.config.hidden_size
        self.num_features = len(self.out_features)

        self.feat_channels = self.num_features * self.hidden_size

        if self.use_feature_projection:
            self.feature_projection = torch.nn.Linear(
                self.hidden_size, self.feature_projection_dim, bias=False
            )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        # Transform images to the right format
        inputs = self.image_processor(images=x, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        # Extract features from the DINOv2 model
        outputs = self.dino(**inputs, output_hidden_states=True)
        hidden_states: tuple[torch.Tensor, ...] = outputs.hidden_states
        hidden_states = tuple(
            (
                hs[:, 5:, : self.feature_projection_dim]
                if not self.use_feature_projection
                else self.feature_projection(hs[:, 5:, :])
            )
            for i, hs in enumerate(hidden_states)
            if i in [2, 5, 8, 11]
        )  # Skip CLS + Register tokens

        return hidden_states
