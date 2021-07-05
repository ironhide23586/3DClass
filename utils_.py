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

MAX_TRAIN_IN_POINTS = 3096

BATCH_SIZE = 4

UPDATE_TENSORBOARD_EVERY_N_STEPS = 10

BASE_LR = 1e-4
NUM_TRAIN_STEPS = 1000000
LR_EXP_DECAY_POWER = .068

LAZ_SCALE_CONST = 4e-5
GT_SCALE = .2
QUANTIZATION_RESOLUTION = .0001
GT_STRIDE = 10
BATCHES_PER_EPOCH = 50

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
import cv2
from plyfile import PlyElement, PlyData
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point

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
new_colors = np.array([(244, 35, 231), (106, 142, 35), (190, 153, 153), (69, 69, 69), (0, 0, 142)])
colorhash_newid_map = {name_colorhash_map[n]: labelname_id_remap[n] for n in included_labels}
n_classes = len(new_labels)

k = 0.015210282150733894
scale_fnames = ['sg27_station1_intensity_rgb.txt', 'sg27_station2_intensity_rgb.txt',
                'sg27_station4_intensity_rgb.txt', 'sg27_station5_intensity_rgb.txt',
                'sg27_station9_intensity_rgb.txt', 'sg28_station4_intensity_rgb.txt']
scale_map = dict(zip(scale_fnames, k / np.array([0.019111323459149544, 0.015210282150733894,
                                                 0.008469265037180073, 0.009127502076506722,
                                                 0.006896646850301384, 0.03305238803503553])))


def force_makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def vector_angle(a, b):
    inner = np.inner(a, b)
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    cos = inner / norms
    rad = np.arccos(np.clip(cos, -1.0, 1.0))
    deg = np.rad2deg(rad)
    return deg


def compare_planarity(plane_points, p):
    v = np.abs(np.linalg.det(plane_points - p))
    return v


def get_planarity_score(p__):
    p_ = np.unique(p__, axis=0)
    if p_.shape[0] < 4:
        return 0.
    tlp = p_.min(axis=0)
    ii_ = np.linalg.norm(tlp - p_, axis=-1).argmin()
    reordered_points = [p_[ii_]]
    buff = np.vstack([p_[:ii_], p_[ii_ + 1:]])
    plane_scores = []
    ref_poly = None
    for ii in range(1, p_.shape[0]):
        ii_ = np.linalg.norm(reordered_points[-1] - buff, axis=-1).argmin()
        reordered_points.append(buff[ii_])
        if ref_poly is None and len(reordered_points) > 3:
            ref_poly = reordered_points[-4:-1]
        if ref_poly is not None:
            plane_scores.append(compare_planarity(ref_poly, reordered_points[-1]))
        if ii + 1 < p_.shape[0]:
            buff = np.vstack([buff[:ii_], buff[ii_ + 1:]])
    plane_scores = np.array(plane_scores) * 1e5
    plane_score = 0
    if plane_scores.shape[0] > 1 and plane_scores.max() != 0:
        plane_score = 1. - plane_scores.mean()
    return plane_score


def reorder_points(p_):
    tlp = p_.min(axis=0)
    pi_ = np.arange(p_.shape[0])
    ii_ = np.linalg.norm(tlp - p_, axis=-1).argmin()
    reordered_points = [p_[ii_]]
    reordered_points_idx = [pi_[ii_]]
    buff = np.vstack([p_[:ii_], p_[ii_ + 1:]])
    buffi = np.hstack([pi_[:ii_], pi_[ii_ + 1:]])
    for ii in range(1, p_.shape[0]):
        ii_ = np.linalg.norm(reordered_points[-1] - buff, axis=-1).argmin()
        reordered_points.append(buff[ii_])
        reordered_points_idx.append(buffi[ii_])
        if ii + 1 < p_.shape[0]:
            buff = np.vstack([buff[:ii_], buff[ii_ + 1:]])
            buffi = np.hstack([buffi[:ii_], buffi[ii_ + 1:]])
    return np.array(reordered_points), np.array(reordered_points_idx)


def point_walk(p_, tlp, stop_thresh=.031):
    reordered_points = []
    reordered_points_idx = []
    pi_ = np.arange(p_.shape[0])
    ds = np.linalg.norm(tlp - p_, axis=-1)
    ii_ = ds.argmin()
    if ds[ii_] > stop_thresh:
        return reordered_points, reordered_points_idx
    reordered_points = [p_[ii_]]
    reordered_points_idx = [pi_[ii_]]
    buff = np.vstack([p_[:ii_], p_[ii_ + 1:]])
    buffi = np.hstack([pi_[:ii_], pi_[ii_ + 1:]])
    for ii in range(1, p_.shape[0]):
        ds = np.linalg.norm(reordered_points[-1] - buff, axis=-1)
        ii_ = ds.argmin()
        if ds[ii_] > stop_thresh:
            break
        reordered_points.append(buff[ii_])
        reordered_points_idx.append(buffi[ii_])
        if ii + 1 < p_.shape[0]:
            buff = np.vstack([buff[:ii_], buff[ii_ + 1:]])
            buffi = np.hstack([buffi[:ii_], buffi[ii_ + 1:]])
    return reordered_points, reordered_points_idx


def scores2labels(scores, tile_xyzs, t=.045):
    labels = scores.argmax(axis=1)
    idx = np.arange(scores.shape[0])
    i = idx[labels == 1]
    p = tile_xyzs[i]

    xs, ys, zs = p.T
    microcube_side = .1
    # cube = []
    nx = int((xs.max() - xs.min()) / microcube_side)
    ny = int((ys.max() - ys.min()) / microcube_side)
    nz = int((zs.max() - zs.min()) / microcube_side)
    xmin, ymin, zmin = p.min(axis=0)
    for y in range(ny):
        for x in range(nx):
            for z in range(nz):
                sx = xmin + x * microcube_side
                sy = ymin + y * microcube_side
                sz = zmin + z * microcube_side
                ex = sx + microcube_side
                ey = sy + microcube_side
                ez = sz + microcube_side
                filt = np.logical_and(
                    np.logical_and(np.logical_and(np.logical_and(np.logical_and(xs >= sx, xs < ex), ys >= sy), ys < ey),
                                   zs >= sz), zs < ez)
                p_ = p[filt]
                if p_.shape[0] > 2:
                    # cube.append([(sx, sy, sz), p_])
                    plane_score = get_planarity_score(p_)
                    if plane_score > .5:
                        chunk_indices = i[filt]
                        chunk_xyzs = tile_xyzs[chunk_indices]
                        labels[chunk_indices] = 3  # building
                        for j in range(chunk_xyzs.shape[0]):
                            xyz = chunk_xyzs[j]
                            # j_ = chunk_indices[j]
                            bag = p[labels[i] == 1]
                            bagi = i[labels[i] == 1]
                            if bag.shape[0] == 0:
                                break
                            _, j__ = point_walk(bag, xyz, t)
                            paint_idx = bagi[j__]
                            labels[paint_idx] = 3  # building
                        # k = 0
                    # # colors = np.tile([[0, (1. - plane_score) * 255, plane_score * 255]], [p_.shape[0], 1])
                    # ls = np.round(plane_score).astype(np.int)
                    # colors = np.tile([[0, (1. - ls) * 255, ls * 255]], [p_.shape[0], 1])
                    # write_ply('scratchspace/g-' + '-'.join(map(str, [plane_score, x, y, z])) + '.ply', p_, colors)
    canvas = np.zeros([GRID_H, GRID_W])
    xys = (((p[:, :2] + 1) / 2.) * [GRID_W, GRID_H]).astype(np.int)
    canvas[xys[:, 1], xys[:, 0]] = (np.clip(p[:, -1] - p[:, -1].min(), 0., 1.) * 255).astype(np.uint8)
    tree_hm = cv2.erode(canvas, np.ones([2, 2]))
    xys = np.unique(np.rollaxis(np.array(np.meshgrid(np.arange(GRID_W),
                                                     np.arange(GRID_H))), 0, 3)[tree_hm > 0], axis=0)
    if xys.shape[0] > 0:
        xys_ = ((xys / [GRID_W, GRID_H]) - .5) * 2.
        d = np.array([np.linalg.norm(p[:, :2] - p_, axis=1) for p_ in xys_])
        d_thresh = 500 * POINT_TILER_SIDE / GRID_W
        d_xys_painted = np.rollaxis(np.array(np.meshgrid(np.arange(d.shape[1]),
                                                         np.arange(d.shape[0]))), 0, 3)[
            np.logical_and(d < d_thresh, d > 0)]
        tree_idx = i[d_xys_painted[:, 0]]
        labels[tree_idx][tile_xyzs[tree_idx, -1] < .36] = 2
        ret = cv2.findContours(cv2.dilate(canvas, np.ones([3, 3])).astype(np.uint8),
                               cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(ret) == 3:
            _, contours, _ = ret
        else:
            contours, _ = ret
        # Find the convex hull object for each contour
        hull_list = []
        for i_ in range(len(contours)):
            a = cv2.contourArea(contours[i_])
            if a < 500 and a > 300:
                hull = cv2.convexHull(contours[i_])
                hull_list.append(np.squeeze(hull))
        points = [Point(p) for p in p[:, :2]]
        f_ = p[:, -1] < .4
        for hull in hull_list:
            hull_xys = (hull / GRID_W - .5) * 2
            poly = Polygon(hull_xys)
            f = np.logical_and(np.array([poly.contains(p) for p in points]), f_)
            car_idx = i[f]
            labels[car_idx] = 4
    labels[tile_xyzs[:, -1] < .15] = 0
    return labels


def sample_data_worker(point_data, random_transform=False):
    ret = point_data.sample_tile()
    if ret is None:
        return None, None
    if len(ret) == 2:
        tile_xyzs, tile_rgbs = ret
        labels = np.expand_dims(rgb2label(tile_rgbs), 0).astype(np.int)
    else:
        tile_xyzs, _, _ = ret
        labels = None
    xyzs = xyz_preprocess(tile_xyzs)
    if random_transform and np.random.random() > .5:  # random 3D rotation
        # angle_x, angle_y, angle_z = np.random.uniform(0, 360, size=3)
        quaternion = np.random.uniform(0, 1, size=4)
        R = get_rot_mat(quaternion)
        xyzs = np.dot(R, xyzs.T).T
    tile_xyzs_normalized = np.expand_dims(xyzs, 0)
    return tile_xyzs_normalized, labels


def sample_data(point_data, random_transform=False):
    xs = []
    ys = []
    cnt = 0
    if random_transform:
        while True:
            x, y = sample_data_worker(point_data, random_transform)
            if x is None:
                continue
            xs.append(x)
            ys.append(y)
            cnt += 1
            if cnt == BATCH_SIZE:
                break
        xs = np.vstack(xs)
        ys = np.vstack(ys)
    else:
        xs, ys = sample_data_worker(point_data, random_transform)
    return xs, ys


def get_rot_mat(quaternion_):
    quaternion = quaternion_ / np.linalg.norm(quaternion_)
    a, b, c, d = quaternion
    rotation_matrix = np.array([[2 * b * c - 2 * a * d, 2 * a ** 2 - 1 + 2 * c ** 2, 2 * c * d + 2 * a * b],
                                [2 * b * d + 2 * a * c, 2 * c * d - 2 * a * b, 2 * a ** 2 - 1 + 2 * d ** 2],
                                [2 * a ** 2 - 1 + 2 * b ** 2, 2 * b * c + 2 * a * d, 2 * b * d - 2 * a * c]])
    return rotation_matrix


def xyz_preprocess(xyzs_, translate=False):
    if translate:
        xyzs = xyzs_ - xyzs_.min(axis=0)
    else:
        xyzs = xyzs_
    # cxyz = np.array([GRID_W / 2., GRID_H / 2., GRID_W / 2.])
    c = GRID_D / GRID_W
    cxyz = np.array([POINT_TILER_SIDE / 2., POINT_TILER_SIDE / 2., xyzs[:, -1].min()])
    return (xyzs - cxyz) / np.array([POINT_TILER_SIDE / 2., POINT_TILER_SIDE / 2., c * POINT_TILER_SIDE / 2.])


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


def write_ply(out_ply_fpath, xyzs__, rgbs__=None, stride=1):
    xyzs_ = np.squeeze(xyzs__)
    rgbs_ = rgbs__
    rgbs = rgbs_
    if rgbs__ is not None:
        rgbs_ = np.squeeze(rgbs__)
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
        data = np.array(list(map(tuple, np.hstack([xyzs.astype(np.float32), rgbs.astype(np.float32)]))),
                        dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                               ('blue', 'u1')])
    else:
        data = np.array(list(map(tuple, xyzs.astype(np.float32))), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
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
