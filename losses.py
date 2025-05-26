import warnings
from dataclasses import dataclass

import torch


@dataclass
class LossQueryRecord:
    """
    Class to hold the query record for loss functions.

    Attributes:
        pred_depth (torch.Tensor): Predicted depth map (B, 1, H, W)
        pred_logvar (torch.Tensor): Predicted log variance map (B, 1, H, W)
        gt_depth (torch.Tensor): Ground truth depth map (B, 1, H, W)
        eps (float): Small value to avoid division by zero
        valid_mask (torch.Tensor): Valid mask for the depth map (B, 1, H, W)
        penalty_weight (float): Weight for the penalty term
    """

    pred_depth: torch.Tensor
    pred_logvar: torch.Tensor
    gt_depth: torch.Tensor

    eps: float = 1e-6
    valid_mask: torch.Tensor = None
    penalty_weight: float = 1


class BaseLoss(torch.nn.Module):
    """
    Base class for all loss functions.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query_record: LossQueryRecord) -> torch.Tensor:
        raise NotImplementedError(
            "Forward method not implemented. Should be implemented by subclass."
        )

    def __str__(self):
        return self.__class__.__name__


class L1Loss(BaseLoss):
    """
    L1 Loss function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query_record: LossQueryRecord) -> torch.Tensor:
        pred_depth = query_record.pred_depth
        gt_depth = query_record.gt_depth

        return torch.nn.functional.l1_loss(pred_depth, gt_depth, reduction="mean")


class L2Loss(BaseLoss):
    """
    L2 Loss function.
    """

    def __init__(self):
        super().__init__()

    def forward(self, query_record: LossQueryRecord) -> torch.Tensor:
        pred_depth = query_record.pred_depth
        gt_depth = query_record.gt_depth

        return torch.nn.functional.mse_loss(pred_depth, gt_depth, reduction="mean")


class SiLogLoss(BaseLoss):
    def __init__(self, lambd=0.5, zero_punish=1e6):
        super().__init__()
        self.lambd = lambd
        self.zero_punish = zero_punish

        # raise Warning("SiLogLoss has not been rigorously tested. Use with caution!")
        warnings.warn(
            "SiLogLoss has not been rigorously tested. Use with caution!",
            UserWarning,
        )

    def forward(
        self,
        query_record: LossQueryRecord,
    ):
        pred_depth = query_record.pred_depth
        gt_depth = query_record.gt_depth
        eps = query_record.eps
        valid_mask = query_record.valid_mask
        penalty_weight = query_record.penalty_weight

        zero_punish_loss = torch.relu(eps - pred_depth).sum()

        # Avoids log(0)
        pred = torch.clamp(pred_depth, min=eps)
        target = torch.clamp(gt_depth, min=eps)

        if valid_mask is not None:
            valid_mask = valid_mask.detach()
            pred = pred[valid_mask]
            target = target[valid_mask]

        diff_log = torch.log(target) - torch.log(pred)
        loss = torch.sqrt(
            torch.pow(diff_log, 2).mean() - self.lambd * torch.pow(diff_log.mean(), 2)
        )
        return (
            loss
            + penalty_weight * torch.log(torch.abs(pred.mean() - target.mean()))
            + zero_punish_loss * self.zero_punish
        )


class NLLLoss(BaseLoss):
    """
    Negative Log Likelihood Loss assuming prior Gaussian distribution on depth.
    """

    def __init__(self):
        # @TODO: Maybe add regularization term and variance penalty term
        super().__init__()

    def forward(self, query_record: LossQueryRecord) -> torch.Tensor:
        """
        :param query_record: LossQueryRecord object containing the predicted depth, predicted log variance, and ground truth depth.
        """

        pred_depth = query_record.pred_depth
        pred_logvar = query_record.pred_logvar
        gt_depth = query_record.gt_depth

        loss = (
            0.5 * torch.exp(-pred_logvar) * (pred_depth - gt_depth) ** 2
            + 0.5 * pred_logvar
        )
        return loss.mean()


class NLLLogLoss(BaseLoss):
    """
    Negative Log Likelihood Loss assuming prior log Gaussian distribution on depth.
    """

    def __init__(self):
        # @TODO: Maybe add regularization term and variance penalty term
        super().__init__()

    def forward(self, query_record: LossQueryRecord):
        eps = query_record.eps
        pred_depth = torch.clamp(query_record.pred_depth, min=eps)
        gt_depth = torch.clamp(query_record.gt_depth, min=eps)
        pred_logvar = query_record.pred_logvar

        diff_log = torch.log(gt_depth) - torch.log(pred_depth)

        loss = 0.5 * torch.exp(-pred_logvar) * diff_log**2 + 0.5 * pred_logvar
        return loss.mean()


class NLLSiLogLoss(BaseLoss):
    """
    Negative Log Likelihood Loss assuming SiLogLoss in place of Euclidean distance.
    """

    def __init__(self, lambd=0.5, zero_punish=1e6):
        # @TODO: Maybe add regularization term and variance penalty term
        super().__init__()
        self.silogloss = SiLogLoss(lambd, zero_punish)

    def forward(self, query_record: LossQueryRecord):
        pred_logvar = query_record.pred_logvar

        diff_log = self.silogloss(query_record)

        loss = 0.5 * torch.exp(-pred_logvar) * diff_log**2 + 0.5 * pred_logvar
        return loss.mean()
