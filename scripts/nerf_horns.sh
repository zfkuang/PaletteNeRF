#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 python main_nerf.py \
../data/nerf_llff_data/horns \
--workspace nerf_horns \
--bound 4 \
--preload \
--fp16 \
--dt_gamma 0 \
--cuda_ray
#--upsample_steps 128 \

# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 24 --dt_gamma 0 --bg_radius 32 -O --gui