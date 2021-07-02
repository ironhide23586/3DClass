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

import os
from multiprocessing import Pool, cpu_count
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np

import utils_
from data_io import LAZElem, PlyElem
from pointnet.model import PointNet

LAZ_FPATH = utils_.DIR + os.sep + 'C_37EZ1_3_2.laz'

laz_data = LAZElem(LAZ_FPATH)
laz_data.load()
ret = laz_data.sample_tile([22, 1])
tile_xyzs_raw, t_vec, _ = ret
tile_xyzs = utils_.xyz_preprocess(tile_xyzs_raw)
utils_.write_ply('tmp.ply', tile_xyzs)
k = 0

# pnet = PointNet(mode='infer')
# pnet.load_weights(utils_.DIR + '/trained_models-0/aerial-pointnet-weights.803-0.36.hdf5')
# pnet.load_weights(utils_.DIR + '/trained_models/aerial-pointnet-weights.06-0.30.hdf5')


def clf_pcl(xy):
    suffix = '-'.join(map(str, xy))
    ret = laz_data.sample_tile(xy)
    if ret is None:
        return
    tile_xyzs_raw, t_vec, _ = ret
    tile_xyzs = utils_.xyz_preprocess(tile_xyzs_raw)
    tile_pred = pnet.infer(tile_xyzs).argmax(axis=-1)
    labels_pred_rgb = utils_.new_colors[tile_pred]
    fp = out_dir + os.sep + prefix + '_' + suffix + '.ply'
    utils_.write_ply(fp, tile_xyzs_raw + t_vec, labels_pred_rgb)


if __name__ == '__main__':
    # pf = PlyElem('C_37EZ1_3_2_22-1.ply')
    # pf.load()
    # xyzs = utils_.xyz_preprocess(pf.xyzs, translate=True)
    # scores = pnet.infer(xyzs, rescale_scores=False)
    #
    # labels = scores.argmax(axis=1)
    #
    # idx = np.arange(scores.shape[0])
    # shrub_idx = idx[labels == 2]
    # shrub_scores = scores[shrub_idx]
    # car_idx = shrub_idx[shrub_scores[:, utils_.labelname_id_remap['cars']] >= .4]
    # labels[car_idx] = utils_.labelname_id_remap['cars']
    #
    # labels_pred_rgb = utils_.new_colors[labels]
    # fp = pf.fname.replace('.ply', '-reclf.ply')
    # utils_.write_ply(fp, pf.xyzs, labels_pred_rgb)

    nx, ny = laz_data.num_xy_tiles
    tile_centers = np.rollaxis(np.array(np.meshgrid(np.arange(nx), np.arange(ny))), 0, 3).reshape([-1, 2])

    prefix = LAZ_FPATH.split(os.sep)[-1].replace('.laz', '')
    out_dir = LAZ_FPATH + '-outs'
    utils_.force_makedir(out_dir)

    for tc in tile_centers:
        clf_pcl(tc)

    p = Pool(cpu_count())
    p.map(clf_pcl, tile_centers)

    k = 0

    # suffix = '-'.join(map(str, [laz_data.start_tile_x, laz_data.start_tile_y]))
    # tile_xyzs_raw = laz_data.sample_tile()
    # while tile_xyzs_raw is not None:
    #     tile_xyzs = utils_.xyz_preprocess(tile_xyzs_raw)
    #     tile_pred = pnet.infer(tile_xyzs).argmax(axis=-1)
    #
    #     labels_pred_rgb = utils_.new_colors[tile_pred]
    #     fp = out_dir + os.sep + prefix + '_' + suffix + '.ply'
    #     utils_.write_ply(fp, tile_xyzs_raw + laz_data.t_vec, labels_pred_rgb)
    #
    #     suffix = '-'.join(map(str, [laz_data.start_tile_x, laz_data.start_tile_y]))
    #     tile_xyzs_raw = laz_data.sample_tile()

    k = 0

    #
    # xyzs = ps.read_laz(LAZ_FPATH)
    #
    # shift_point = xyzs_.min(axis=0)
    # xyzs = xyzs_ - shift_point
    # scale_point = xyzs.max(axis=0)
    # xyzs = xyzs / scale_point
    #
