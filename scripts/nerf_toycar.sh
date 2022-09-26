#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py \
../data/refnerf/toycar \
--workspace nerf_toycar \
--offset 0 -0.6 0 \
--bg_radius 32 \
--preload \
--fp16 \
--cuda_ray \
--test 
# --dt_gamma 0 \

# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 24 --dt_gamma 0 --bg_radius 32 -O --gui