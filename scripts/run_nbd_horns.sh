#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nbd.py ../data/nerf_llff_data/horns/ \
--workspace horns_nbd \
--fp16 \
--preload \
--num_basis 5 \
--roughness_list 0.9 \
--viewdir_roughness \
--color_cluster_num 5 \
# --use_initialization_from_rays \
# --test
# --optimize_roughness \