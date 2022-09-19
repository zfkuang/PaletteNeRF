#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nbd.py ../data/nerf_llff_data/horns/ \
--workspace horns_nbd \
--fp16 \
--preload \
--num_basis 15 \
--roughness_list 1.0 \
--optimize_roughness \
--lambda_sparsity 0.003 \
--guidance \
--max_guide_epoch 50 \
# --use_initialization_from_rays \