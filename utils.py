import torch


def compute_metrics(
    pred_depth: torch.Tensor, gt_depth: torch.Tensor
) -> dict[str, float]:
    B = pred_depth.size(0)

    abs_diff = torch.abs(pred_depth - gt_depth)
    mae = torch.sum(abs_diff).item()
    rmse = torch.sum(torch.pow(abs_diff, 2)).item()
    rel = torch.sum(abs_diff / (gt_depth + 1e-6)).item()

    sirmse = 0.0

    # Calculate scale-invariant RMSE for each image in the batch
    for i in range(B):
        pred = pred_depth[i].squeeze()
        gt = gt_depth[i].squeeze()

        EPSILON = 1e-6

        valid_gt = gt > EPSILON
        if not torch.any(valid_gt):
            continue

        gt_valid = gt[valid_gt]
        pred_valid = pred[valid_gt]

        log_gt = torch.log(gt_valid)

        pred_valid = torch.where(pred_valid > EPSILON, pred_valid, EPSILON)
        log_pred = torch.log(pred_valid)

        # Calculate scale-invariant error
        diff = log_pred - log_gt
        diff_mean = torch.mean(diff)

        # Calculate RMSE for this image
        sirmse += torch.sqrt(torch.mean((diff - diff_mean) ** 2)).item()

    # Calculate thresholded accuracy
    max_ratio = torch.max(
        pred_depth / (gt_depth + 1e-6), gt_depth / (pred_depth + 1e-6)
    )
    delta1 = torch.sum(max_ratio < 1.25).item()
    delta2 = torch.sum(max_ratio < 1.25**2).item()
    delta3 = torch.sum(max_ratio < 1.25**3).item()

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "siRMSE": sirmse,
        "REL": rel,
        "Delta1": delta1,
        "Delta2": delta2,
        "Delta3": delta3,
        "num_samples": B,
        "num_pixels": B * pred_depth.size(2) * pred_depth.size(3),
    }

    return metrics
