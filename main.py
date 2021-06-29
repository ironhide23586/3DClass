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


DIR = 'scratchspace'
LAZ_FPATH = DIR + os.sep + 'C_37EZ1_3_2.laz'


GT_PCL_FPATH = 'scratchspace/sg28_station5_xyz_intensity_rgb/sg28_station5_xyz_intensity_rgb.txt'
PLY_FPATH = 'scratchspace/C_37EZ1_3_2.ply'

if __name__ == '__main__':
    ps = PointStreamer()

    xyzs, rgbs = ps.read_ply(PLY_FPATH)

    if rgbs is None:
        zs_ = xyzs[:, -1]
        zs = np.clip(zs_, -600, 120)
        m = zs - zs.min()
        m = m / m.max()
        m = np.tile(np.expand_dims(m, -1), [1, 3])
        rgbs = (m * [0, 255, 0] + (1 - m) * [0, 0, 255]).astype(np.uint8)
        # import matplotlib.pyplot as plt
        # plt.hist(zs[np.logical_and(zs > -600, zs < -120)], 100)
        # plt.show()
    xyzs = xyzs - xyzs.min(axis=0)
    xyzs = xyzs / xyzs.max()
    ps.write_ply(xyzs, rgbs, PLY_FPATH.replace('.ply', '_depth.ply'))

    # a = ps.get_points()
    las = pylas.read(LAZ_FPATH)
    xyzs_ = np.vstack([las['X'], las['Y'], las['Z']]).T
    ps.write_ply(xyzs_, fpath=LAZ_FPATH.replace('.laz', '.ply'))

    shift_point = xyzs_.min(axis=0)
    xyzs = xyzs_ - shift_point
    scale_point = xyzs.max(axis=0)
    xyzs = xyzs / scale_point

