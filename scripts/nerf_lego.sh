#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py \
../data/nerf_synthetic/lego \
--workspace nerf_lego \
-O \
--scale 0.8 \
--dt_gamma 0
