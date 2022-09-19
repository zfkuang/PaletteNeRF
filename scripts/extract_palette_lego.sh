#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_palette.py \
../data/nerf_synthetic/lego \
./results/nerf_lego/version_1/checkpoints/ngp_ep0300.pth \
-O \
--bound 1 \
--scale 0.8 \
--dt_gamma 0 \
