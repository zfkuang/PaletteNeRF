#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
../data/nerf_llff_data/horns \
./results/nerf_horns/version_1/checkpoints/ngp_ep0556.pth \
-O \
--bound 4 \
--scale 0.16 \
--offset 0 0 1.5 \
--extract_palette \


# python main_palette.py ../data/nerf_synthetic/lego ./results/nerf_lego/version_4/checkpoints/ngp_ep0300.pth --fp16 --preload --bound 1 --scale 0.8 --dt_gamma 0 --extract_palette --test