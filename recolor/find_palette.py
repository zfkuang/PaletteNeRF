import os
import random
import numpy as np

from PIL import Image
from sklearn.cluster import KMeans

import rgbsg
import plenoctree as plnoct

import time
from typing import Tuple


OUTPUT_DIR_ROOT = './output'

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
    model_path: str,
    tau: float = 8e-3,
    palette_size: int = 6
):
    assert palette_size >= 4 ## convex hull should have at least 4 vertices
    print(f'extracting palette with {palette_size} colors')

    assert os.path.basename(model_path).split('.')[1] == 'npz'
    model_name = os.path.basename(model_path).split('.')[0]

    output_dir = os.path.join(OUTPUT_DIR_ROOT, model_name)
    if not os.path.exists(output_dir):
        print(f'create output directory {output_dir}')
        os.makedirs(output_dir)

    output_prefix = os.path.join(output_dir, model_name)
    print('output_prefix:', output_prefix)

    def get_log_elapsed(start: float, stage: str) -> float:
        elapsed = time.time() - start
        assert elapsed > 0
        print(f'timing ({model_name}): {stage} took {elapsed:.3f} sec.')
        return elapsed

    total_elapsed = 0

    ## radiance sampling
    start = time.time()

    tree = plnoct.PlenOctree(model_path)
    dirs = plnoct.generate_dirs_cylindrical(2)
    colors_raw, colors, weights = tree.sample_radiance(dirs)
    print('radiance samples:', colors_raw.shape, colors.shape, weights.shape)
    colors_raw = colors_raw.reshape(-1,3)
    colors = colors.reshape(-1,3)
    weights = weights.flatten()

    total_elapsed += get_log_elapsed(start, stage='radiance sampling')

    assert len(weights[weights < 0]) == 0, 'negative weight indicates the failure of radiance sampling'


    ## save nosy raw radiance samples (outside timing analysis)
    res = 800
    n_total = res**2
    random.seed(0)
    idcs = random.sample(range(len(colors_raw)), n_total)
    assert len(idcs) == len(set(idcs)), 'each element of idcs should be unique'
    img = colors_raw[idcs].reshape(res,res,3)
    Image.fromarray((img*255).round().clip(0,255).astype(np.uint8)).save(output_prefix+"-radiance-raw.png")


    ## radiance sample filtering
    start = time.time()

    ## coarse histogram (2^3 = 8 bins)
    bin_weights_coarse, bin_centers_coarse = plnoct.compute_RGB_histogram(colors, weights, bits_per_channel=3)
    sum_weights = np.sum(bin_weights_coarse)
    bin_weights_coarse /= sum_weights

    idcs = bin_weights_coarse > tau
    bin_weights_coarse = bin_weights_coarse[idcs]
    bin_centers_coarse = bin_centers_coarse[idcs]

    ## fine histogram (2^5 = 8 bins)
    bin_weights_fine, bin_centers_fine = plnoct.compute_RGB_histogram(colors, weights, bits_per_channel=5)
    idcs = bin_weights_fine > 0
    bin_weights_fine = bin_weights_fine[idcs]
    bin_weights_fine /= sum_weights
    bin_centers_fine = bin_centers_fine[idcs]

    centers, center_weights = run_kmeans(
        n_clusters=len(bin_weights_coarse), points=bin_centers_fine,
        init=bin_centers_coarse, sample_weight=bin_weights_fine)

    total_elapsed += get_log_elapsed(start, stage='sample filtering')


    ## convex hull simplification
    start = time.time()

    palette_rgb = rgbsg.Hull_Simplification_posternerf(
        centers.astype(np.double), output_prefix,
        pixel_counts=center_weights,
        target_size=palette_size)

    total_elapsed += get_log_elapsed(start, stage='hull simplification')


    ## total extraction time
    print(f'timing ({model_name}): palette extraction (total) took {total_elapsed:.3f} sec.')


    ## save palette
    palette_img = rgbsg.get_bigger_palette_to_show(palette_rgb)
    Image.fromarray((palette_img*255).round().clip(0,255).astype(np.uint8)).save(output_prefix+"-palette.png")

    rgbsg.write_palette_txt(palette_rgb, output_prefix+'-palette.txt')
    np.savez(os.path.join(output_dir, 'palette.npz'), palette=palette_rgb)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('tree_npz', help='path to the plenoctree npz file')
    args = parser.parse_args()

    palette_extraction(args.tree_npz)
