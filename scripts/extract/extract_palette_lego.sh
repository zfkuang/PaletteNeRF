#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=4 python main_palette.py \
../data/nerf_synthetic/lego \
./results/nerf_lego/version_3/checkpoints/ngp_ep0300.pth \
-O \
--scale 0.8 \
--dt_gamma 0 \
--extract_palette \


# python main_palette.py ../data/nerf_synthetic/lego ./results/nerf_lego/version_4/checkpoints/ngp_ep0300.pth --fp16 --preload --bound 1 --scale 0.8 --dt_gamma 0 --extract_palette --test