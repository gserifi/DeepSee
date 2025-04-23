#!/bin/bash

srun \
--account=$1 \
--ntasks=1 \
--cpus-per-task=8 \
--gpus=rtx_4090:1 \
--time=4:00:00 \
--job-name="interactive" \
--mem-per-cpu=4096 \
--pty bash -l