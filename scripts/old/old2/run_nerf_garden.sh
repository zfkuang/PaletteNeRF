#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=3 python main_nerf.py ../data/mip360/garden/ \
--workspace garden_nerf \
-O \
--scale 1.0 \
--bound 5.0 \
--dt_gamma 0 \
# --test
# --use_initialization_from_rays \