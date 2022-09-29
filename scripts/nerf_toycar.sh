#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_nerf.py \
../data/refnerf/toycar \
--workspace nerf_toycar \
--bound 4 \
--preload \
--fp16 \
--cuda_ray \
--dt_gamma 0 \
--iters 150000 \
--version_id 6

# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 4 --dt_gamma 0 --bg_radius 32 -O --gui