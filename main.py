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
from data_io import PlySet

DIR = 'scratchspace'
GT_DIR = DIR + os.sep + 'training_data'

LAZ_FPATH = DIR + os.sep + 'C_37EZ1_3_2.laz'
PLY_FPATH = DIR + os.sep + 'C_37EZ1_3_2.ply'


if __name__ == '__main__':
    # ply_fps = glob(GT_DIR + os.sep + '*rgb.ply')
    # point_data = PlySet(ply_fps)
    # point_data.match_scales()
    #
    # tile_xyzs, tile_rgbs = point_data.sample_point_tile()
    # utils_.write_ply('tmp.ply', tile_xyzs, tile_rgbs, stride=1)
    # grid_x, grid_y = utils_.points2grid(tile_xyzs, tile_rgbs)

    xyzs, rgbs = utils_.read_ply(PLY_FPATH)
    rgbs = utils_.z2rgb(xyzs[:, -1])

    xyzs_unique = npi.unique(xyzs)
    i_ = npi.indices(xyzs, xyzs_unique)
    xyzs = xyzs[i_]
    rgbs = rgbs[i_]

    utils_.write_ply(PLY_FPATH.replace('.ply', '_depth.ply'),
                     np.floor(utils_.normalize_xyzs(xyzs * utils_.LAZ_SCALE_CONST,
                                                    scale_const=1. / utils_.GRID_RESOLUTION)),
                     rgbs)
    k = 0

    #
    # xyzs = ps.read_laz(LAZ_FPATH)
    #
    # shift_point = xyzs_.min(axis=0)
    # xyzs = xyzs_ - shift_point
    # scale_point = xyzs.max(axis=0)
    # xyzs = xyzs / scale_point
    #
