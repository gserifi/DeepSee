#!/bin/bash

CONFIGS=(pdpt_mse_default pdpt_nll_cross_all_feat_feat pdpt_nll_cross_all_feat_image pdpt_nll_cross_all_image_feat pdpt_nll_cross_all_image_image pdpt_nll_default pdpt_nll_residual pdpt_nllsilogloss_default pretrained_depth_anything_v2 ptdec_image ptdec unet)

for config in "${CONFIGS[@]}"; do
    sbatch \
    --account=$1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --gpus=rtx_4090:1 \
    --time=24:00:00 \
    --job-name="$config" \
    --mem-per-cpu=2048 \
    --wrap="echo $config; python3 main.py fit -c configs/ablations/$config.yaml"
done