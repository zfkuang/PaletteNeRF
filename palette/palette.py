import os
import random
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Function
from torch.cuda.amp import custom_bwd, custom_fwd

from PIL import Image
from sklearn.cluster import KMeans

import time
from typing import Tuple
from .rgbsg import *

try:
    import _nbd_palette as _backend
except ImportError:
    print("Loading module nbd_palette...")
    from .backend import _backend

def compute_RGB_histogram(
    colors_rgb: np.ndarray,
    weights: np.ndarray,
    bits_per_channel: int
) -> Tuple[np.ndarray, np.ndarray]:
    assert colors_rgb.ndim == 2 and colors_rgb.shape[1] == 3
    assert weights.ndim == 1
    assert len(colors_rgb) == len(weights)
    assert 1 <= bits_per_channel and bits_per_channel <=8

    try:
        bin_weights, bin_centers_rgb = _backend.compute_RGB_histogram(
            colors_rgb.flatten(), weights.flatten(), bits_per_channel)
    except RuntimeError as err:
        print(err)
        assert False

    return bin_weights, bin_centers_rgb

def run_kmeans(
    n_clusters: int,
    points: np.ndarray,
    init: np.ndarray,
    sample_weight: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    print(f'running kmeans with K = {n_clusters}')
    kmeans = KMeans(n_clusters=n_clusters, init=init).fit(X=points, sample_weight=sample_weight)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    center_weights = np.zeros(n_clusters)
    for i in range(n_clusters):
        center_weights[i] = np.sum(sample_weight[labels==i])

    idcs = np.argsort(center_weights * -1)

    return centers[idcs], center_weights[idcs]

def palette_extraction(
    inputs: dict,
    output_dir: str,
    tau: float = 8e-3,
    palette_size: int = 6
):
    assert palette_size >= 4 ## convex hull should have at least 4 vertices
    print(f'extracting palette with {palette_size} colors')

    if not os.path.exists(output_dir):
        print(f'create output directory {output_dir}')
        os.makedirs(output_dir)
    
    output_prefix = "%s/extract"%output_dir
    ## radiance sampling
    start = time.time()

    colors = inputs['colors']
    weights = np.ones_like(colors[...,0])
    colors = colors.reshape(-1,3)
    weights = weights.flatten()

    assert len(weights[weights < 0]) == 0, 'negative weight indicates the failure of radiance sampling'

    ## save radiance samples (outside timing analysis)
    res = 800
    n_total = res**2
    random.seed(0)
    idcs = random.sample(range(len(colors)), n_total)
    assert len(idcs) == len(set(idcs)), 'each element of idcs should be unique'
    img = colors[idcs].reshape(res,res,3)
    Image.fromarray((img*255).round().clip(0,255).astype(np.uint8)).save(output_prefix+"-radiance-raw.png")

    ## radiance sample filtering
    start = time.time()

    ## coarse histogram (2^3 = 8 bins)
    bin_weights_coarse, bin_centers_coarse = compute_RGB_histogram(colors, weights, bits_per_channel=3)
    sum_weights = np.sum(bin_weights_coarse)
    bin_weights_coarse /= sum_weights

    idcs = bin_weights_coarse > tau
    bin_weights_coarse = bin_weights_coarse[idcs]
    bin_centers_coarse = bin_centers_coarse[idcs]

    ## fine histogram (2^5 = 8 bins)
    bin_weights_fine, bin_centers_fine = compute_RGB_histogram(colors, weights, bits_per_channel=5)
    idcs = bin_weights_fine > 0
    bin_weights_fine = bin_weights_fine[idcs]
    bin_weights_fine /= sum_weights
    bin_centers_fine = bin_centers_fine[idcs]

    centers, center_weights = run_kmeans(
        n_clusters=len(bin_weights_coarse), points=bin_centers_fine,
        init=bin_centers_coarse, sample_weight=bin_weights_fine)

    ## convex hull simplification
    start = time.time()

    palette_rgb = Hull_Simplification_posternerf(
        centers.astype(np.double), output_prefix,
        pixel_counts=center_weights,
        target_size=palette_size)

    ## save palette
    palette_img = get_bigger_palette_to_show(palette_rgb)
    Image.fromarray((palette_img*255).round().clip(0,255).astype(np.uint8)).save(output_prefix+"-palette.png")

    write_palette_txt(palette_rgb, output_prefix+'-palette.txt')
    np.savez(os.path.join(output_dir, 'palette.npz'), palette=palette_rgb)
