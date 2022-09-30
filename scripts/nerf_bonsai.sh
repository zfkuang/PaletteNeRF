#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main_nerf.py \
../data/mip360/bonsai \
--workspace nerf_bonsai \
--bound 4 \
--preload \
--fp16 \
--dt_gamma 0 \
--upsample_steps 128 \
--iters 150000 \

# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 24 --dt_gamma 0 --bg_radius 32 -O --gui