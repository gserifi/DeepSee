import lightning as lit
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from base_model import ImageAndDepthDatasetItemType, LitBaseModel


class PretrainedDepthAnythingV2(LitBaseModel):
    """
    Uses pretrained Depth-Anything-V2-Base (https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf) for full
    inference. This model should not be trained.
    """

    def __init__(self):
        super().__init__()

        self.image_processor = AutoImageProcessor.from_pretrained(
            "depth-anything/Depth-Anything-V2-Base-hf"
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(
            "depth-anything/Depth-Anything-V2-Base-hf"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Transform images to the right format
        inputs = self.image_processor(images=x, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(x.device) for k, v in inputs.items()}

        # Extract the depth output and interpolate to correct size
        outputs = self.model(**inputs)
        outputs = outputs.predicted_depth

        depth = F.interpolate(
            outputs.unsqueeze(1),
            size=x.shape[2:],
            mode="bicubic",
            align_corners=False,
        )

        return depth

    def training_step(self, batch: ImageAndDepthDatasetItemType, batch_idx: int):
        """
        Model should not be used for training
        """
        raise ValueError("PretrainedDepthAnythingV2 cannot be trained")

    def on_load_checkpoint(self, checkpoint):
        """
        Model is not allowed to load any checkpoints
        """
        raise ValueError(
            "PretrainedDepthAnythingV2 does not support loading checkpoints."
        )
