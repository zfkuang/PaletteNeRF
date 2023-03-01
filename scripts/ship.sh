#! /bin/bash

datatype="blender"
name="nerf_ship"
bound=2
scale=0.8
bg_radius=0
density_thresh=10
iters=30000
offset='0 0 0'
random_size=0
data_dir="../data/nerf_synthetic/ship"
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
    -O \
    --dt_gamma 0 \
    $test_mode
elif [[ $model == 'extract' ]]; then
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
    $data_dir \
    $nerf_model \
    -O \
    --bound ${bound} \
    --scale ${scale} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh}  \
    --extract_palette --use_normalized_palette
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
    --random_size ${random_size} \
    --use_initialization_from_rgbxy \
    --use_normalized_palette \
        --dt_gamma 0 \
    --datatype ${datatype} \
    $test_mode
else
    echo "Invalid model. Options are: nerf, extract, palette"
fi

# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 24 --dt_gamma 0 --bg_radius 32 -O --gui