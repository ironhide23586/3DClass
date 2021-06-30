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

import utils_
from data_io import PlySet

DIR = 'scratchspace'
GT_DIR = DIR + os.sep + 'training_data'

LAZ_FPATH = DIR + os.sep + 'C_37EZ1_3_2.laz'
PLY_FPATH = DIR + os.sep + 'C_37EZ1_3_2.ply'


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


if __name__ == '__main__':
    # convert_gt_to_ply(GT_DIR)

    ply_fps = glob(GT_DIR + os.sep + '*rgb.ply')
    point_data = PlySet(ply_fps)
    point_data.match_scales()
    tile_xyzs, tile_rgbs = point_data.sample_point_tile()
    utils_.write_ply(tile_xyzs, 'tmp.ply', tile_rgbs, stride=1)
    utils_.points2grid(tile_xyzs, tile_rgbs)

    k = 0

    # xyzs, rgbs = utils_.read_ply(PLY_FPATH)
    # rgbs = utils_.z2rgb(xyzs[:, -1])
    #
    # xyzs_unique = npi.unique(xyzs)
    # i_ = npi.indices(xyzs, xyzs_unique)
    # xyzs = xyzs[i_]
    # rgbs = rgbs[i_]
    #
    # utils_.write_ply(utils_.normalize_xyzs(xyzs), rgbs, PLY_FPATH.replace('.ply', '_depth.ply'))


    #
    # xyzs = ps.read_laz(LAZ_FPATH)
    #
    # shift_point = xyzs_.min(axis=0)
    # xyzs = xyzs_ - shift_point
    # scale_point = xyzs.max(axis=0)
    # xyzs = xyzs / scale_point
    #
