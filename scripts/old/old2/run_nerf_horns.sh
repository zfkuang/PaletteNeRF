#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py ../data/nerf_llff_data/horns --workspace nerf_horns -O
