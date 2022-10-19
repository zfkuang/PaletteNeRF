#! /bin/bash

name="nerf_lego"
bound=2
scale=0.8
bg_radius=0
density_thresh=10
iters=30000
data_dir="../data/nerf_synthetic/lego"
nerf_model="./results/${name}/version_1/checkpoints/ngp_ep0300.pth"
dt_gamma="--dt_gamma 0"

while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--test)
      test=True
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
    $dt_gamma \
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
    $dt_gamma
elif [[ $model == 'palette' ]]; then
    OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=0 python main_palette.py \
    $data_dir \
    $nerf_model \
    -O \
    --iters ${iters} \
    --bound ${bound} \
    --scale ${scale} \
    --bg_radius ${bg_radius} \
    --density_thresh ${density_thresh} \
    --use_initialization_from_rgbxy \
    --model_mode palette \
    --use_normalized_palette \
    --separate_radiance \
    $dt_gamma \
    $test_mode
else
    echo "Invalid model. Options are: nerf, extract, palette"
fi

# python main_nerf.py ../data/refnerf/toycar --workspace nerf_toycar --bound 24 --dt_gamma 0 --bg_radius 32 -O --gui