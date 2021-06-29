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

import numpy as np
import pylas

from data_io import PointStreamer
import utils

DIR = 'scratchspace'
LAZ_FPATH = DIR + os.sep + 'C_37EZ1_3_2.laz'
PLY_FPATH = DIR + os.sep + 'C_37EZ1_3_2.ply'

GT_PCL_FPATH = DIR + os.sep + 'sg27_station9_intensity_rgb/sg27_station9_intensity_rgb.txt'
GT_PCL_LABEL_FPATH = DIR + os.sep + 'sem8_labels_training/sg27_station9_intensity_rgb.labels'


if __name__ == '__main__':
    ps = PointStreamer(GT_PCL_FPATH, GT_PCL_LABEL_FPATH)

    # xyzs, rgbs = ps.read_ply(PLY_FPATH)
    # rgbs = utils.z2rgb(xyzs[:, -1])
    # ps.write_ply(utils.normalize_xyzs(xyzs), rgbs, PLY_FPATH.replace('.ply', '_depth.ply'))

    data, labels = ps.get_points(stride=1000, normalize_point_locs=True, scale=.15)
    blend_coeff = 1.
    xyzs = data[:, :3]
    xyzs = xyzs + [.084, .142, .017]
    rgbs = blend_coeff * utils.label_colors[labels] + (1. - blend_coeff) * data[:, 3:]
    ps.write_ply(xyzs, rgbs)

    xyzs = ps.read_laz(LAZ_FPATH)

    shift_point = xyzs_.min(axis=0)
    xyzs = xyzs_ - shift_point
    scale_point = xyzs.max(axis=0)
    xyzs = xyzs / scale_point

