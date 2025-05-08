from pathlib import Path

import lightning as lit
import numpy as np
import torch
from matplotlib import cm
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid

from data_module import ImageAndDepthDatasetItem
from losses import BaseLoss, L2Loss, LossQueryRecord
from utils import compute_metrics

PREDICTIONS_DIR = Path("./data/predictions")


class LitBaseModel(lit.LightningModule):
    def __init__(self, loss_fns: list[tuple[BaseLoss, float]] = None):
        super().__init__()
        self.save_hyperparameters()

        self.loss_fns = loss_fns if loss_fns is not None else [(L2Loss(), 1.0)]
        for loss_fn, _ in self.loss_fns:
            loss_fn.eval()

        self.train_metrics: list[dict[str, float]] = []
        self.val_metrics: list[dict[str, float]] = []
        self.test_metrics: list[dict[str, float]] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "Forward method not implemented. Should be implemented by subclass."
        )

    def log_batch_images(
        self,
        image: torch.Tensor,
        gt_depth: torch.Tensor,
        pred_depth: torch.Tensor,
        pred_logvar: torch.Tensor,
        stage: str,
        num_samples: int = 8,
    ):
        image_grid = make_grid(image[:num_samples], nrow=4, normalize=True)
        gt_depth_grid = make_grid(gt_depth[:num_samples], nrow=4, normalize=True)
        pred_depth_grid = make_grid(pred_depth[:num_samples], nrow=4, normalize=True)

        pred_std = torch.exp(0.5 * pred_logvar[:num_samples])
        pred_std = (pred_std - pred_std.min()) / (
            pred_std.max() - pred_std.min() + 1e-8
        )
        viridis_cmap = cm.get_cmap("viridis")
        pred_std = (
            torch.from_numpy(
                viridis_cmap(
                    pred_std.permute(0, 2, 3, 1).squeeze(-1).detach().cpu().numpy()
                )
            )
            .to(pred_std)
            .permute(0, 3, 1, 2)[:, :3, :, :]
        )
        pred_std_grid = make_grid(pred_std, nrow=4, normalize=True)
        std_opacity = 0.7
        pred_std_grid = std_opacity * pred_std_grid + (1 - std_opacity) * image_grid

        self.logger.experiment.add_image(f"{stage}/image", image_grid, self.global_step)

        self.logger.experiment.add_image(
            f"{stage}/gt_depth", gt_depth_grid, self.global_step
        )

        self.logger.experiment.add_image(
            f"{stage}/pred_depth", pred_depth_grid, self.global_step
        )

        self.logger.experiment.add_image(
            f"{stage}/pred_std", pred_std_grid, self.global_step
        )

    @staticmethod
    def agg_metrics(metrics: list[dict[str, float]], stage: str) -> dict[str, float]:
        total_samples = sum(m["num_samples"] for m in metrics)
        total_pixels = sum(m["num_pixels"] for m in metrics)

        total_mae = sum(m["MAE"] for m in metrics)
        total_rmse = sum(m["RMSE"] for m in metrics)
        total_sirmse = sum(m["siRMSE"] for m in metrics)
        total_rel = sum(m["REL"] for m in metrics)
        total_delta1 = sum(m["Delta1"] for m in metrics)
        total_delta2 = sum(m["Delta2"] for m in metrics)
        total_delta3 = sum(m["Delta3"] for m in metrics)

        pixel_norm = total_samples * total_pixels

        agg_dict = {
            "MAE": total_mae / pixel_norm,
            "RMSE": np.sqrt(total_rmse / pixel_norm),
            "siRMSE": total_sirmse / total_samples,
            "REL": total_rel / pixel_norm,
            "Delta1": total_delta1 / pixel_norm,
            "Delta2": total_delta2 / pixel_norm,
            "Delta3": total_delta3 / pixel_norm,
        }

        return {f"{stage}/{k}": v for k, v in agg_dict.items()}

    def compute_loss(
        self,
        pred_depth: torch.Tensor,
        pred_logvar: torch.Tensor,
        gt_depth: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        total_loss = 0
        log_dict = {}

        query_record = LossQueryRecord(
            pred_depth=pred_depth, pred_logvar=pred_logvar, gt_depth=gt_depth
        )

        for loss_fn, weight in self.loss_fns:
            _loss = loss_fn(query_record)
            total_loss += weight * _loss
            log_dict[str(loss_fn)] = _loss.item()

        return total_loss, log_dict

    def log_losses(
        self, log_dict: dict[str, float], on_step: bool, on_epoch: bool, stage: str
    ):
        for key, value in log_dict.items():
            self.log(
                f"{stage}/{key}",
                value,
                prog_bar=False,
                on_step=on_step,
                on_epoch=on_epoch,
            )

    def training_step(self, batch: ImageAndDepthDatasetItem, batch_idx: int):
        image, gt_depth = batch

        pred_depth, pred_logvar = self.forward(image)

        loss, log_dict = self.compute_loss(pred_depth, pred_logvar, gt_depth)
        self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=False)
        self.log_losses(log_dict, on_step=True, on_epoch=False, stage="train")

        if (self.global_step + 1) % self.trainer.log_every_n_steps == 0:
            self.log_batch_images(
                image, gt_depth, pred_depth, pred_logvar, stage="train"
            )

        self.train_metrics.append(compute_metrics(pred_depth, gt_depth))

        return loss

    def on_train_epoch_end(self):
        agg_metrics = self.agg_metrics(self.train_metrics, stage="train")
        self.log_dict(agg_metrics)

        self.train_metrics.clear()

    def validation_step(
        self,
        batch: ImageAndDepthDatasetItem,
        batch_idx: int,
        skip_log: bool = False,
    ):
        image, gt_depth = batch

        pred_depth, pred_logvar = self.forward(image)

        loss, log_dict = self.compute_loss(pred_depth, pred_logvar, gt_depth)

        self.log("val/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log_losses(log_dict, on_step=False, on_epoch=True, stage="val")

        if batch_idx == 0:
            self.log_batch_images(image, gt_depth, pred_depth, pred_logvar, stage="val")

        self.val_metrics.append(compute_metrics(pred_depth, gt_depth))

        return loss

    def on_validation_epoch_end(self):
        agg_metrics = self.agg_metrics(self.val_metrics, stage="val")
        self.log_dict(agg_metrics)

        self.val_metrics.clear()

    def test_step(self, batch: ImageAndDepthDatasetItem, batch_idx: int):
        image, gt_depth = batch
        pred_depth = self.forward(image)

        self.test_metrics.append(compute_metrics(pred_depth, gt_depth))

    def on_test_epoch_end(self):
        agg_metrics = self.agg_metrics(self.test_metrics, stage="test")

        for metric, value in agg_metrics.items():
            print(f"{metric}: {value}")

        self.test_metrics.clear()

    def predict_step(self, batch: ImageAndDepthDatasetItem, batch_idx: int):
        image, filename = batch
        pred_depth = self.forward(image)

        pred_depth = pred_depth.squeeze(1).cpu()

        PREDICTIONS_DIR.mkdir(exist_ok=True)
        for pred, name in zip(pred_depth, filename):
            depth_im = to_pil_image(make_grid([pred], nrow=1, normalize=True))
            depth_im.save(PREDICTIONS_DIR / name.replace(".npy", ".png"))
            np.save(PREDICTIONS_DIR / name, pred.numpy())

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer
