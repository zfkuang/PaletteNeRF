#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=6 python main_palette.py \
../blender_dataset/kitti \
./results/nerf_kitti/version_2/checkpoints/ngp_ep0300.pth \
--fp16 \
--preload \
--bound 4 \
--scale 3.2 \
--dt_gamma 0 \
--use_initialization_from_rgbxy \
--model_mode palette \
--color_space linear \
--use_normalized_palette \
--multiply_delta 

# python main_palette.py ../data/nerf_synthetic/lego ./results/nerf_lego/version_4/checkpoints/ngp_ep0300.pth --fp16 --preload --bound 1 --scale 0.8 --dt_gamma 0 --extract_palette --test