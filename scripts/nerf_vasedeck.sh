#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=2 python main_nerf.py \
../data/vasedeck \
--workspace nerf_vasedeck \
--bound 2 \
--scale 0.33 \
--dt_gamma 0 \
-O \

# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 24 --dt_gamma 0 --bg_radius 32 -O --gui