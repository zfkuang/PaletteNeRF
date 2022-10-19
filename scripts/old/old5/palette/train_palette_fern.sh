#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
../data/nerf_llff_data/fern \
./results/nerf_fern/version_3/checkpoints/ngp_ep1528.pth \
-O \
--bound 2 \
--scale 0.16 \
--use_initialization_from_rgbxy \
--model_mode palette \
--use_normalized_palette \
--separate_radiance \

# python main_palette.py ../data/nerf_synthetic/lego ./results/nerf_lego/version_4/checkpoints/ngp_ep0300.pth --fp16 --preload --bound 1 --scale 0.8 --dt_gamma 0 --extract_palette --test