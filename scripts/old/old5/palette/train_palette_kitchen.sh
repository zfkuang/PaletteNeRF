#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
../data/mip360/kitchen \
./results/nerf_kitchen/version_2/checkpoints/ngp_ep0369.pth \
-O \
--bound 2 \
--scale 0.12 \
--use_initialization_from_rgbxy \
--model_mode palette \
--use_normalized_palette \
--separate_radiance \
--iters 90000 \
# --pred_clip \
# --sigma_clip 0.5 
# --use_cosine_distance \
# --multiply_delta \

# python main_palette.py ../data/nerf_synthetic/lego ./results/nerf_lego/version_4/checkpoints/ngp_ep0300.pth --fp16 --preload --bound 1 --scale 0.8 --dt_gamma 0 --extract_palette --test