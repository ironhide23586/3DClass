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


import os

import pylas
import numpy as np
from tqdm import tqdm
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

colorhash_name_map = dict(zip(label_color_hashes, label_names))


def z2rgb(zs_, min_z=-600, max_z=120):
    zs = np.clip(zs_, min_z, max_z)
    m = zs - zs.min()
    m = m / m.max()
    m = np.tile(np.expand_dims(m, -1), [1, 3])
    rgbs = (m * [0, 255, 0] + (1 - m) * [0, 0, 255]).astype(np.uint8)
    return rgbs


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


def write_ply(self, xyzs, out_ply_fpath, rgbs=None):
    data = []
    for i in range(xyzs.shape[0]):
        if rgbs is not None:
            data.append((float(xyzs[i][0]), float(xyzs[i][1]), float(xyzs[i][2]),
                         int(rgbs[i][-3]), int(rgbs[i][-2]), int(rgbs[i][-1])))
        else:
            data.append((float(xyzs[i][0]), float(xyzs[i][1]), float(xyzs[i][2])))
    if rgbs is not None:
        v = np.array(data,
                     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                            ('blue', 'u1')])
    else:
        v = np.array(data,
                     dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
    el = PlyElement.describe(v, 'vertex')
    print('Writing points to', self.out_ply_fpath, ' 👉 Number of points =', v.shape[0])
    PlyData([el]).write(out_ply_fpath)


def parse_plydata(ply_data):
    xs = ply_data['vertex']['x']
    ys = ply_data['vertex']['y']
    zs = ply_data['vertex']['z']
    xyzs = np.vstack([xs, ys, zs]).T
    rgbs = None
    print('⚡ Found', xs.shape[0], 'points')
    if 'red' in ply_data['vertex'] and 'green' in ply_data['vertex'] and 'blue' in ply_data['vertex']:
        print('Found RGB data ☢')
        rs = ply_data['vertex']['red']
        gs = ply_data['vertex']['green']
        bs = ply_data['vertex']['blue']
        rgbs = np.vstack([rs, gs, bs]).T
    return xyzs, rgbs


def read_laz(fpath):
    las = pylas.read(fpath)
    xyzs = np.vstack([las['X'], las['Y'], las['Z']]).T
    return xyzs

