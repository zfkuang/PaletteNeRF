#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=9 python main_nbd.py ../data/nerf_synthetic/lego/ \
--workspace lego_nbd \
--fp16 \
--preload \
--num_basis 10 \
--roughness_list 1.0 \
--color_cluster_num 5 \
--use_initialization_from_rays \
--hybrid \
--no_sg \
--sg_num_basis 5 \
--lambda_sparsity 0.0003 \
--lambda_dir 0.001 \
--guidance \
--max_guide_epoch 50 \
# --use_initialization_from_rays \
# --test
# --optimize_roughness \