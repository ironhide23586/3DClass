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

DIR = 'scratchspace'
GT_DIR = DIR + os.sep + 'training_data'

LAZ_FPATH = DIR + os.sep + 'C_37EZ1_3_2.laz'
# PLY_FPATH = DIR + os.sep + 'C_37EZ1_3_2.ply'


if __name__ == '__main__':
    ply_fps = glob(GT_DIR + os.sep + '*rgb.ply')
    point_data = PlySet(ply_fps)
    point_data.match_scales()

    tile_xyzs, tile_rgbs = point_data.sample_tile()
    utils_.write_ply('tmp.ply', tile_xyzs, tile_rgbs, stride=1)
    grid_x_gt, grid_y_gt = utils_.points2grid(tile_xyzs, tile_rgbs)

    laz_data = LAZElem(LAZ_FPATH)
    tile_xyzs = laz_data.sample_tile()
    utils_.write_ply('tmp_laz.ply', tile_xyzs, utils_.z2rgb(tile_xyzs[:, -1]), stride=1)
    grid_x_in, _ = utils_.points2grid(tile_xyzs)

    #
    # xyzs = ps.read_laz(LAZ_FPATH)
    #
    # shift_point = xyzs_.min(axis=0)
    # xyzs = xyzs_ - shift_point
    # scale_point = xyzs.max(axis=0)
    # xyzs = xyzs / scale_point
    #
