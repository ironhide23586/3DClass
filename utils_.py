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

DIR = 'scratchspace'
GT_DIR = DIR + '/training_data'

BASE_LR = 1e-4
NUM_TRAIN_STEPS = 1000000
LR_EXP_DECAY_POWER = .068

LAZ_SCALE_CONST = 4e-5
GT_SCALE = .2
QUANTIZATION_RESOLUTION = .0001
GT_STRIDE = 10
BATCHES_PER_EPOCH = 100

MAX_Z = .025
POINT_TILER_SIDE = .12

GRID_H = 256
GRID_W = 256
GRID_D = 120

GRID_RESOLUTION = POINT_TILER_SIDE / GRID_W

from glob import glob
import os

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
included_labels = [l for l in label_names if l not in excluded_labels]
colorhash_name_map = dict(zip(label_color_hashes, label_names))
name_colorhash_map = dict(zip(label_names, label_color_hashes))
excluded_label_hashes = [name_colorhash_map[n] for n in excluded_labels]

labelname_id_remap = {'man-made-terrain': 0, 'natural-terrain': 0, 'hard-scape': 0,
                      'high-vegetation': 1, 'low-vegetation': 2, 'buildings': 3, 'cars': 4}
new_labels = ['ground-surface', 'high-vegetation', 'low-vegetation', 'buildings', 'cars']
colorhash_newid_map = {name_colorhash_map[n]: labelname_id_remap[n] for n in included_labels}
n_classes = len(new_labels)


def force_makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def sample_data(point_data):
    tile_xyzs, tile_rgbs = point_data.sample_tile()
    labels = rgb2label(tile_rgbs)
    cxyz = np.array([GRID_W / 2., GRID_H / 2., GRID_W / 2.])
    tile_xyzs_normalized = (tile_xyzs - cxyz) / cxyz
    return np.expand_dims(tile_xyzs_normalized, 0), np.expand_dims(labels, 0)


def z2rgb(zs_, min_z=-600, max_z=120):
    zs = np.clip(zs_, min_z, max_z)
    m = zs - zs.min()
    m = m / m.max()
    m = np.tile(np.expand_dims(m, -1), [1, 3])
    rgbs = (m * [0, 255, 0] + (1 - m) * [0, 0, 255]).astype(np.uint8)
    return rgbs


def rgb2label(tile_rgbs):
    labels = color2hash(tile_rgbs)
    for h in colorhash_newid_map:
        labels[labels == h] = colorhash_newid_map[h]
    return labels


def points2grid(xyzs, rgbs=None):
    xs, ys, zs = xyzs.T.astype(np.int)
    canvas_in = np.zeros([GRID_H, GRID_W, GRID_D]).astype(np.float)
    canvas_in[ys, xs, zs] = 1.
    canvas_label = None
    if rgbs is not None:
        labels = color2hash(rgbs)
        for h in colorhash_newid_map:
            labels[labels == h] = colorhash_newid_map[h]
        canvas_label = np.zeros([GRID_H, GRID_W, GRID_D]).astype(np.int)
        canvas_label[ys, xs, zs] = labels
    return canvas_in, canvas_label


def blend_data(data, labels, blend_coeff=1.):
    xyzs = data[:, :3] + [.084, .142, .017]
    rgbs = blend_coeff * label_colors[labels] + (1. - blend_coeff) * data[:, 3:]
    return xyzs, rgbs


def normalize_xyzs(xyzs_, translate_const=None, scale_const=None):
    if translate_const is None:
        translate_const = xyzs_.min(axis=0)
    xyzs = xyzs_ - translate_const
    if scale_const is None:
        scale_const = 1. / xyzs.max()
    xyzs = xyzs * scale_const
    return xyzs


def read_ply(fpath, raw=False):
    print('ψ(｀∇´)ψ Reading', fpath)
    ply_data = PlyData.read(fpath)
    if raw:
        return ply_data['vertex']
    xyzs, rgbs = parse_plydata(ply_data['vertex'])
    return xyzs, rgbs


def write_ply(out_ply_fpath, xyzs_, rgbs_=None, stride=1):
    if stride > 1:
        idx = np.arange(0, xyzs_.shape[0], stride)
        xyzs = xyzs_[idx]
        if rgbs_ is not None:
            rgbs = rgbs_[idx]
    else:
        xyzs = xyzs_
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


def convert_gt_to_ply(gt_dir):  # WARNING: Takes a LONG time (╯°□°）╯︵ ┻━┻
    from data_io import PointStreamer, PlyIO
    all_gt_pcl_fpaths = glob(gt_dir + os.sep + 'raw_data/*.txt')
    all_gt_label_fpaths = [fp.replace('.txt', '.labels') for fp in all_gt_pcl_fpaths]
    n_fpaths = len(all_gt_pcl_fpaths)
    point_streamers = []
    plyios = []
    for i in range(n_fpaths):
        point_streamers.append(PointStreamer(all_gt_pcl_fpaths[i], all_gt_label_fpaths[i]))
    for i in range(n_fpaths):
        plyios.append(PlyIO(point_streamers[i]))
    for i in range(n_fpaths):
        plyios[i].dump_ply()
