from typing import Any

import lightning as lit
from torch import nn


class LitBaseModel(lit.LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.loss_fn = nn.MSELoss()
        self.loss_fn.eval()

    def forward(self, x):
        raise NotImplementedError("Forward method not implemented")

    def training_step(self, batch, batch_idx):
        image, gt_depth = batch
        pred_depth = self.forward(image)

        loss = self.loss_fn(pred_depth, gt_depth)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch, batch_idx):
        image, gt_depth = batch
        pred_depth = self.forward(image)

        loss = self.loss_fn(pred_depth, gt_depth)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def test_step(self, batch, batch_idx):
        # TODO: check what the submission requires
        image, _ = batch
        pred_depth = self.forward(image)
        return pred_depth
