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
from glob import glob

import numpy_indexed as npi
import numpy as np

import utils_
from data_io import PlySet, LAZElem
from pointnet.model import PointNet


# LAZ_FPATH = DIR + os.sep + 'C_37EZ1_3_2.laz'
# PLY_FPATH = DIR + os.sep + 'C_37EZ1_3_2.ply'


if __name__ == '__main__':
    ply_fps = glob(utils_.GT_DIR + os.sep + '*rgb.ply')

    pnet = PointNet(mode='train')
    pnet.load_data(ply_fps[:-1], [ply_fps[-1]])
    pnet.load_weights(utils_.DIR + '/trained_models-0/aerial-pointnet-weights.803-0.36.hdf5')

    # from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    #
    # xyzs, labels_gt = utils_.sample_data(pnet.val_point_data)
    # labels_pred = pnet.model.predict(xyzs).argmax(axis=-1)
    # acc = accuracy_score(labels_gt[0], labels_pred[0])
    # prec, prec, fsc, _ = precision_recall_fscore_support(labels_gt[0], labels_pred[0], labels=[0, 1, 2, 3, 4, 5])
    #
    # labels_pred_rgb = utils_.new_colors[labels_pred]
    # utils_.write_ply('pred.ply', xyzs[0], labels_pred_rgb[0])

    pnet.train()


    # point_data = PlySet(ply_fps)
    # point_data.match_scales()
    #
    # tile_xyzs, tile_rgbs = point_data.sample_tile()
    # utils_.write_ply('tmp.ply', tile_xyzs, tile_rgbs, stride=1)
    # grid_x_gt, grid_y_gt = utils_.points2grid(tile_xyzs, tile_rgbs)

    # laz_data = LAZElem(LAZ_FPATH)
    # tile_xyzs = laz_data.sample_tile()
    # utils_.write_ply('tmp_laz.ply', tile_xyzs, utils_.z2rgb(tile_xyzs[:, -1]), stride=1)
    # grid_x_in, _ = utils_.points2grid(tile_xyzs)

    k = 0

    #
    # xyzs = ps.read_laz(LAZ_FPATH)
    #
    # shift_point = xyzs_.min(axis=0)
    # xyzs = xyzs_ - shift_point
    # scale_point = xyzs.max(axis=0)
    # xyzs = xyzs / scale_point
    #
