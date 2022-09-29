#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_nerf.py \
../data/nerf_llff_data/horns \
--workspace nerf_horns \
--bound 4 \
--preload \
--fp16 \
--upsample_steps 128 \
--dt_gamma 0 \

# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 24 --dt_gamma 0 --bg_radius 32 -O --gui