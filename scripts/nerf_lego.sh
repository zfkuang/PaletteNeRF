#! /bin/bash

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py ../data/nerf_synthetic/lego --workspace nerf_lego --bound 1 --scale 0.8 --dt_gamma 0 --preload --fp16
