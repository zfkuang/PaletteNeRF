#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=7 python main_nbd.py ../blender_dataset/kitti/ \
--workspace kitti_nbd_hybrid \
--fp16 \
--preload \
--bound 1 \
--scale 0.8 \
--dt_gamma 0 \
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
--test \
--ckpt 'latest'
# --single_radiance \
# --optimize_roughness \