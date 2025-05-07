import torch
import torch.nn as nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5, zero_punish=1e6):
        super().__init__()
        self.lambd = lambd
        self.zero_punish = zero_punish

        raise Warning("SiLogLoss has not been rigorously tested. Use with caution!")

    def forward(
        self, pred_orig, target_orig, eps=1e-6, valid_mask=None, penalty_weight=1
    ):

        zero_punish_loss = torch.relu(eps - pred_orig).sum()

        # Avoids log(0)
        pred = torch.clamp(pred_orig, min=eps)
        target = torch.clamp(target_orig, min=eps)

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


class NLLoss(nn.Module):
    """
    Negative Log Likelihood Loss assuming prior Gaussian distribution on depth.
    """

    def __init__(self):
        # @TODO: Maybe add regularization term and variance penalty term
        super().__init__()

    def forward(self, pred_depth, pred_logvar, gt_depth):
        """
        :param pred_depth: Predicted depth map (B, 1, H, W)
        :param  pred_logvar: Predicted log variance map (B, 1, H, W)
        :param gt_depth: Ground truth depth map (B, 1, H, W)
        """
        loss = (
            0.5 * torch.exp(-pred_logvar) * (pred_depth - gt_depth) ** 2
            + 0.5 * pred_logvar
        )
        return loss.mean()
