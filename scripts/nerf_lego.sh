#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main_nerf.py \
../data/nerf_synthetic/lego \
--workspace nerf_lego \
--bound 4 \
--scale 3.2 \
--dt_gamma 0 \
--preload \
--fp16
