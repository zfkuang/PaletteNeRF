#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=9 python main_nbd.py ../data/nerf_llff_data/room/ \
--workspace room_nbd \
--fp16 \
--preload \
--num_basis 12 \
--roughness_list 1.0 \
--color_cluster_num 6 \
--use_initialization_from_rays \
--hybrid \
--no_sg \
--sg_num_basis 6 \
--lambda_sparsity 0.0003 \
--lambda_dir 0.001 \
--guidance \
--max_guide_epoch 50 \
# --use_initialization_from_rays \
# --test
# --optimize_roughness \