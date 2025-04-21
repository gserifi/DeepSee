from typing import Any

import lightning as lit
from torch import nn
from torchvision.utils import make_grid


class LitBaseModel(lit.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.MSELoss()
        self.loss_fn.eval()

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented")

    def log_batch_images(self, image, gt_depth, pred_depth, stage, n=8):
        image_grid = make_grid(image[:n], nrow=4, normalize=True)
        gt_depth_grid = make_grid(gt_depth[:n], nrow=4, normalize=True)
        pred_depth_grid = make_grid(pred_depth[:n], nrow=4, normalize=True)

        self.logger.experiment.add_image(f"{stage}/image", image_grid, self.global_step)

        self.logger.experiment.add_image(
            f"{stage}/gt_depth", gt_depth_grid, self.global_step
        )

        self.logger.experiment.add_image(
            f"{stage}/pred_depth", pred_depth_grid, self.global_step
        )

    def training_step(self, batch, batch_idx):
        image, gt_depth = batch
        pred_depth = self.forward(image)

        loss = self.loss_fn(pred_depth, gt_depth)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        if self.global_step % 10 == 0:
            self.log_batch_images(image, gt_depth, pred_depth, stage="train")

        return loss

    def validation_step(self, batch, batch_idx):
        image, gt_depth = batch
        pred_depth = self.forward(image)

        loss = self.loss_fn(pred_depth, gt_depth)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)

        if batch_idx == 0:
            self.log_batch_images(image, gt_depth, pred_depth, stage="val")

        return loss

    def test_step(self, batch, batch_idx):
        # TODO: check what the submission requires
        image, _ = batch
        pred_depth = self.forward(image)
        return pred_depth
