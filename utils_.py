"""  _
    |_|_
   _  | |
 _|_|_|_|_
|_|_|_|_|_|_
  |_|_|_|_|_|
    | | |_|
    |_|_
      |_|
Author: Souham Biswas
Website: https://www.linkedin.com/in/souham/
"""


GT_SCALE = .2
QUANTIZATION_RESOLUTION = .0001
GT_STRIDE = 10

MAX_Z = .025
POINT_TILER_SIDE = .12

GRID_H = 256
GRID_W = 256
GRID_D = 120

GRID_RESOLUTION = POINT_TILER_SIDE / GRID_W

# from multiprocessing import Pool, cpu_count


import pylas
import numpy as np
from plyfile import PlyElement, PlyData

label_colors = [(0, 0, 0), (244, 35, 231), (152, 250, 152),
                # 0 = unlabelled, 1 = man-made-terrain, 2 = natural-terrain
                (106, 142, 35), (190, 153, 153), (69, 69, 69),
                # 3 = high-vegetation, 4 = low-vegetation, 5 = buildings,
                (128, 64, 128), (255, 255, 255), (0, 0, 142)]
# 6 = hard-scape, 7 = scan-artifacts, 8 = cars
label_colors = np.array(label_colors)

color2hash = lambda rgb_colors: np.array(rgb_colors)[:, 2]

label_color_hashes = color2hash(label_colors)

label_names = ['unlabelled', 'man-made-terrain', 'natural-terrain', 'high-vegetation', 'low-vegetation',
               'buildings', 'hard-scape', 'scan-artifacts', 'cars']
excluded_labels = ['unlabelled', 'scan-artifacts', 'hard-scape']
colorhash_name_map = dict(zip(label_color_hashes, label_names))
name_colorhash_map = dict(zip(label_names, label_color_hashes))
excluded_label_hashes = [name_colorhash_map[n] for n in excluded_labels]


def z2rgb(zs_, min_z=-600, max_z=120):
    zs = np.clip(zs_, min_z, max_z)
    m = zs - zs.min()
    m = m / m.max()
    m = np.tile(np.expand_dims(m, -1), [1, 3])
    rgbs = (m * [0, 255, 0] + (1 - m) * [0, 0, 255]).astype(np.uint8)
    return rgbs


def points2grid(xyzs, rgbs):
    xs, ys, zs = xyzs.T.astype(np.int)
    color_hashes = color2hash(rgbs)
    canvas_in = np.zeros([GRID_H, GRID_W, GRID_D]).astype(np.float)
    canvas_label = np.ones([GRID_H, GRID_W, GRID_D]).astype(np.int)
    canvas_in[ys, xs, zs] = 1.

    import cv2
    cv2.imwrite('a.png', canvas_in.max(axis=-1) * 255)


def blend_data(data, labels, blend_coeff=1.):
    xyzs = data[:, :3] + [.084, .142, .017]
    rgbs = blend_coeff * label_colors[labels] + (1. - blend_coeff) * data[:, 3:]
    return xyzs, rgbs


def normalize_xyzs(xyzs_, translate_const=None, scale_const=None):
    if translate_const is None:
        translate_const = xyzs_.min(axis=0)
    xyzs = xyzs_ - translate_const
    if scale_const is None:
        scale_const = xyzs.max()
    xyzs = xyzs * scale_const
    return xyzs


def read_ply(fpath, raw=False):
    print('ψ(｀∇´)ψ Reading', fpath)
    ply_data = PlyData.read(fpath)
    if raw:
        return ply_data['vertex']
    xyzs, rgbs = parse_plydata(ply_data)
    return xyzs, rgbs


def write_ply(xyzs_, out_ply_fpath, rgbs_=None, stride=1):
    if stride > 1:
        idx = np.arange(0, xyzs_.shape[0], stride)
        xyzs = xyzs_[idx]
        if rgbs_ is not None:
            rgbs = rgbs_[idx]
    else:
        xyzs = xyzs_
        if rgbs_ is not None:
            rgbs = rgbs_
    print('Writing points to', out_ply_fpath, ' 👉 Number of points =', xyzs.shape[0])
    if rgbs is not None:
        data = np.array(list(map(tuple, np.hstack([xyzs, rgbs]))),
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                               ('blue', 'u1')])
    else:
        data = np.array(list(map(tuple, xyzs)), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(data, 'vertex')
    PlyData([el]).write(out_ply_fpath)
    print('👌 done! ☜(ﾟヮﾟ☜)')


def parse_plydata(ply_data):
    xs = ply_data['x']
    ys = ply_data['y']
    zs = ply_data['z']
    xyzs = np.vstack([xs, ys, zs]).T
    rgbs = None
    print('⚡ Found', xs.shape[0], 'points')
    if 'red' in ply_data and 'green' in ply_data and 'blue' in ply_data:
        print('Found RGB data ☢')
        rs = ply_data['red']
        gs = ply_data['green']
        bs = ply_data['blue']
        rgbs = np.vstack([rs, gs, bs]).T
    return xyzs, rgbs


def read_laz(fpath):
    las = pylas.read(fpath)
    xyzs = np.vstack([las['X'], las['Y'], las['Z']]).T
    return xyzs
