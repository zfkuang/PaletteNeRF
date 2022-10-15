#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=5 python main_palette.py \
../data/mip360/kitchen \
./results/nerf_kitchen/version_2/checkpoints/ngp_ep0369.pth \
-O \
--bound 2 \
--scale 0.12 \
--use_initialization_from_rgbxy \
--model_mode palette \
--multiply_delta \
--iters 90000 \
--use_normalized_palette \
--use_cosine_distance \
# --pred_clip \
# --test \


# python main_palette.py ../data/nerf_synthetic/lego ./results/nerf_lego/version_4/checkpoints/ngp_ep0300.pth --fp16 --preload --bound 1 --scale 0.8 --dt_gamma 0 --extract_palette --test