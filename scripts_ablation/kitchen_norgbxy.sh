#! /bin/bash

datatype="mip360"
name="nerf_kitchen"
bound=2 
scale=0.12 
bg_radius=0
offset='0 0 0'
density_thresh=10
lambda_sparse=0.01
iters=90000
min_near=0.05
random_size=0
data_dir='../data/mip360/kitchen'
nerf_model=./results/${name}/version_1
ablation_name="nerf_kitchen_norgbxy"

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--test)
      test=True
      shift # past argument
      shift # past value
      ;;    
    -v|--video)
      video=True
      shift # past argument
      shift # past value
      ;;
    -g|--gui)
      gui=True
      shift # past argument
      shift # past value
      ;;
    -m|--model)
      model="$2"
      shift # past argument
      shift # past value
      ;;
  esac
done

if [ $gui ]; then
    test_mode='--test --gui'
elif [ $video ]; then
    test_mode='--test --video'
elif [ $test ]; then
    test_mode='--test'
else
    test_mode=''
fi

OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
$data_dir \
$nerf_model \
--iters ${iters} \
--bound ${bound} \
--scale ${scale} \
--offset ${offset} \
--bg_radius ${bg_radius} \
--density_thresh ${density_thresh} \
--min_near ${min_near} \
--random_size ${random_size} \
--model_mode palette \
--use_normalized_palette \
--separate_radiance \
--datatype ${datatype} \
$test_mode \
-O \
--max_freeze_palette_epoch 0 \
--lambda_weight 0 \
--num_basis 5 \
--ablation_name ${ablation_name} \
#--dt_gamma 0 \


# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 24 --dt_gamma 0 --bg_radius 32 -O --gui