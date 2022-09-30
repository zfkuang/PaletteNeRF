#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_nerf.py \
../blender_dataset/kitti \
--workspace nerf_kitti \
--bound 4 \
--scale 3.2 \
--dt_gamma 0 \
--upsample_steps 128 \
--preload \
--fp16
