#! /bin/bash

datatype="llff"
name="nerf_flower"
bound=2
scale=0.02
bg_radius=0
offset='0 0 1.5'
density_thresh=10
lambda_sparse=0.05
iters=10000
min_near=0.2
random_size=0
data_dir='../data/nerf_llff_data/flower'
nerf_model=./results/${name}/version_1

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

if [[ $model == 'nerf' ]]; then
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_nerf.py \
    $data_dir \
    --workspace ${name} \
    --iters ${iters} \
    --bound ${bound} \
    --offset ${offset} \
    --scale ${scale} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh} \
    --min_near ${min_near} \
    --no_bg \
    -O \
    $test_mode
elif [[ $model == 'extract' ]]; then
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
    $data_dir \
    $nerf_model \
    -O \
    --bound ${bound} \
    --scale ${scale} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh} \
    --min_near ${min_near} \
    --extract_palette
elif [[ $model == 'palette' ]]; then
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
    $data_dir \
    $nerf_model \
    -O \
    --iters ${iters} \
    --bound ${bound} \
    --scale ${scale} \
    --offset ${offset} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh} \
    --min_near ${min_near} \
    --random_size ${random_size} \
    --use_initialization_from_rgbxy \
    --model_mode palette \
    --use_normalized_palette \
    --separate_radiance \
    --datatype ${datatype} \
    $test_mode
else
    echo "Invalid model. Options are: nerf, extract, palette"
fi

# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 24 --dt_gamma 0 --bg_radius 32 -O --gui