#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py \
../data/mip360/room \
--workspace nerf_room \
--bound 2 \
--scale 0.12 \
--iters 90000 \
-O \

# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 24 --dt_gamma 0 --bg_radius 32 -O --gui