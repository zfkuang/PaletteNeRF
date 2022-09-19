#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nbd.py ../data/mip360/garden/ \
--workspace garden_nbd \
--fp16 \
--preload \
--scale 1.0 \
--bound 5.0 \
--dt_gamma 0 \
--num_basis 15 \
--roughness_list 1.0 \
--optimize_roughness \
--lambda_sparsity 0.003 \
# --test
# --use_initialization_from_rays \