#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nbd.py ../blender_dataset/kitti/ \
--workspace kitti_nbd_nosg \
--fp16 \
--preload \
--bound 1 \
--scale 0.8 \
--dt_gamma 0 \
--num_basis 7 \
--roughness_list 0.9 \
--viewdir_roughness \
--color_cluster_num 7 \
--no_sg 
# --use_initialization_from_rays \
# --test
# --optimize_roughness \