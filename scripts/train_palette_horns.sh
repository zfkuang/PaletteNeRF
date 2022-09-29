#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
../data/nerf_llff_data/horns \
./results/nerf_horns/version_5/checkpoints/ngp_ep0556.pth \
--fp16 \
--preload \
--bound 4 \
--dt_gamma 0 \
--use_initialization_from_rgbxy \
--model_mode palette \
--use_normalized_palette \
--color_space linear

# python main_palette.py ../data/nerf_synthetic/lego ./results/nerf_lego/version_4/checkpoints/ngp_ep0300.pth --fp16 --preload --bound 1 --scale 0.8 --dt_gamma 0 --extract_palette --test