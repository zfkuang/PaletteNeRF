from .fastLayerDecomposition import Additive_mixing_layers_extraction as Tan18
Tan18.DEMO = True

import numpy as np

from scipy.spatial import ConvexHull


def get_bigger_palette_to_show(palette):
    ##### palette shape is M*3
    c=50
    palette2=np.ones((1*c, len(palette)*c, 3))
    for i in range(len(palette)):
        palette2[:,i*c:i*c+c,:]=palette[i,:].reshape((1,1,-1))
    return palette2

# adapted from Tan18.Hull_Simplification_determined_version
# assuming data is in range (0,1)
def Hull_Simplification_posternerf(
    data, output_prefix,
    pixel_counts=None, error_thres=2.0/255.0, target_size=None
) -> np.ndarray:
    data = data.reshape(-1,3)
    hull = ConvexHull(data)
    origin_vertices = hull.points[hull.vertices]
    print("original hull vertices number:", len(hull.vertices))
    
    output_rawhull_obj_file=output_prefix+"-mesh_obj_files.obj"
    Tan18.write_convexhull_into_obj_file(hull, output_rawhull_obj_file)

    if pixel_counts is not None:
        assert len(data) == len(pixel_counts)
        print("use specified pixel_counts as weights")
        unique_data = data
    else:
        print("computing unique pixels and their counts")
        unique_data, pixel_counts=Tan18.get_unique_colors_and_their_counts(data)
        print("number of unique pixels:", len(unique_data))
    

    max_loop=5000
    for i in range(max_loop):
        if i % 10  == 0:
            print("loop:", i)
        mesh=Tan18.TriMesh.FromOBJ_FileName(output_rawhull_obj_file)
        old_num=len(mesh.vs)
        old_vertices=mesh.vs
        mesh=Tan18.remove_one_edge_by_finding_smallest_adding_volume_with_test_conditions(mesh,option=2)
        hull=ConvexHull(mesh.vs)
        Tan18.write_convexhull_into_obj_file(hull, output_rawhull_obj_file)

        if len(hull.vertices) <= 10:
            if target_size is None:
                reconstruction_errors=Tan18.outsidehull_points_distance_unique_data_version(hull.points[ hull.vertices ].clip(0.0,1.0), unique_data, pixel_counts)
                print('reconstruction_erros:', reconstruction_errors)

                if reconstruction_errors>error_thres:
                    oldhull=ConvexHull(old_vertices)
                    Tan18.write_convexhull_into_obj_file(oldhull, output_prefix + '-org-final.obj')
                    return oldhull.points[ oldhull.vertices ].clip(0.0,1.0).reshape(-1,3)

            elif len(hull.vertices) == target_size:
                Tan18.write_convexhull_into_obj_file(hull, output_prefix + '-org-final.obj')
                return hull.points[ hull.vertices ].clip(0.0,1.0).reshape(-1,3)
    
        if len(hull.vertices)==old_num or len(hull.vertices)==4:
            Tan18.write_convexhull_into_obj_file(hull, output_prefix + '-org-final.obj')
            return hull.points[ hull.vertices ].clip(0.0,1.0).reshape(-1,3)

    print('hull simplification failed')
    return origin_vertices.clip(0.0,1.0).reshape(-1,3)


def read_palette_txt(path):
    palette = []
    with open(path, 'r') as f:
        palette = [np.array([float(a) for a in line.split(' ')]) for line in f.readlines()]
    return np.vstack(palette)

def write_palette_txt(palette_rgb, path):
    palette_str = ''
    for i, c in enumerate(palette_rgb):
        palette_str += f'{c[0]} '
        palette_str += f'{c[1]} '
        if i < len(palette_rgb) - 1:
            palette_str += f'{c[2]}\n'
        else:
            palette_str += f'{c[2]}'
    with open(path, 'w') as f:
        f.write(palette_str)


## the palette extraction method of [Tan et al. 16]
def Tan16_palette_extraction(colors_rgb: np.ndarray, output_prefix: str, target_size: int) -> np.ndarray:
    return Hull_Simplification_posternerf(colors_rgb, output_prefix, target_size=target_size)


## the palette extraction method of [Chao et al. 21]
def Chao21_palette_extraction(
    colors_rgb: np.ndarray,
    output_prefix: str,
    target_size: int,
    K: int = 40
) -> np.ndarray:
    colors_rgb = colors_rgb.reshape(-1,3)

    ## for very large input images (e.g., concatenation of several images)
    original_count = len(colors_rgb)
    unique_colors, pixel_counts = Tan18.get_unique_colors_and_their_counts(colors_rgb)
    print(f'found {len(unique_colors)} unique colors out of {original_count}')

    print(f'running kmeans with K = {K}')
    kmeans = KMeans(n_clusters=K, random_state=0).fit(unique_colors, sample_weight=pixel_counts)

    centers_rgb = kmeans.cluster_centers_
    center_weights = np.array([np.sum(pixel_counts[kmeans.labels_==i]) for i in range(K)])
    np.savez(output_prefix+'-centers.npz', centers=centers_rgb)

    assert np.sum(center_weights) == original_count

    palette_rgb = Hull_Simplification_posternerf(
        centers_rgb, pixel_counts=center_weights,
        output_prefix=output_prefix,
        target_size=target_size)
    
    return palette_rgb


if __name__ == '__main__':
    import argparse
    import os
    from PIL import Image
    from sklearn.cluster import KMeans

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img', help='source png file')
    parser.add_argument('-d', '--dir', help='directory containing srouce png files')
    parser.add_argument('-k', '--n_clusters', type=int, default=40, help='k of k-means in the method of [Chao et al. 21]')
    args = parser.parse_args()

    img = None
    if args.img is not None:
        img = np.asfarray(Image.open(args.img).convert('RGB'))/255.0
    elif args.dir is not None:
        imgs = [(np.asfarray(Image.open(os.path.join(args.dir, file)).convert('RGB'))/255.0).reshape(-1,3)
            for file in os.listdir(args.dir)]
        print('concatenate {} images'.format(len(imgs)))
        img = np.vstack(imgs)

    if img is not None:
        if not os.path.exists('./tmp'):
            os.makedirs('./tmp')
        output_prefix = './tmp/test'

        palette_rgb = Chao21_palette_extraction(
            img, output_prefix, target_size=6, K=args.n_clusters)

        palette_img = get_bigger_palette_to_show(palette_rgb)
        Image.fromarray((palette_img*255).round().astype(np.uint8)).save(output_prefix+'-palette.png')

        np.savez(output_prefix+'-palette.npz', palette=palette_rgb)
        write_palette_txt(palette_rgb, path=output_prefix+'-palette.txt')
    else:
        print('failed to load input image')
