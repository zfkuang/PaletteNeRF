#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nbd.py ../data/nerf_synthetic/lego/ \
--workspace lego_nbd \
--fp16 \
--preload \
--bound 1 \
--scale 0.8 \
--dt_gamma 0 \
--num_basis 5 \
--roughness_list 0.9 \
--viewdir_roughness \
--color_cluster_num 5 \
--use_initialization_from_rays \
# --test
# --optimize_roughness \