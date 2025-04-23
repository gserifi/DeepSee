#!/bin/bash

sbatch \
--account=$1 \
--ntasks=1 \
--cpus-per-task=8 \
--gpus=rtx_4090:1 \
--time=4:00:00 \
--job-name="$2" \
--mem-per-cpu=4096 \
--mail-type=END \
--wrap="python3 main.py predict -c configs/$2.yaml --ckpt_path $3"