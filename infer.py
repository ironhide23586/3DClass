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

from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np

import utils_
from data_io import LAZElem
from pointnet.model import PointNet

# LAZ_FPATH = utils_.DIR + os.sep + 'C_37EZ1_0_2.laz'


def clf_pcl(xy, pnet, out_dir, prefix, laz_data):
    suffix = '-'.join(map(str, xy))
    fp = out_dir + os.sep + prefix + '_' + suffix + '.ply'
    if os.path.exists(fp):
        print(fp, 'exists, skipping... (╯°□°）╯︵ ┻━┻')
    print('(☞ﾟヮﾟ)☞ Processing', fp)
    ret = laz_data.sample_tile(xy)
    if ret is None:
        return
    tile_xyzs_raw, t_vec, _ = ret
    tile_xyzs = utils_.xyz_preprocess(tile_xyzs_raw)
    if tile_xyzs.shape[0] > 12000:
        i = []
        rgbs = []
        pidx = np.arange(tile_xyzs.shape[0])
        n_passes = int(np.ceil(tile_xyzs.shape[0] / 12000))
        print('Big PCL, running', n_passes, 'passes sampling random points uniformly in each')
        filt = np.ones(tile_xyzs.shape[0]).astype(np.bool)
        for k_ in range(n_passes):
            if k_ < n_passes - 1:
                idx = np.random.choice(pidx[filt], 12000)
                filt[idx] = False
            else:
                idx = pidx[filt]
            tile_pred = utils_.scores2labels(pnet.infer(tile_xyzs[idx]), tile_xyzs[idx])
            labels_pred_rgb = utils_.new_colors[tile_pred]
            i.append(idx)
            rgbs.append(labels_pred_rgb)
        i_ = np.hstack(i)
        utils_.write_ply(fp, tile_xyzs_raw[i_] + t_vec, np.vstack(rgbs))
    else:
        tile_pred = utils_.scores2labels(pnet.infer(tile_xyzs), tile_xyzs)
        labels_pred_rgb = utils_.new_colors[tile_pred]
        utils_.write_ply(fp, tile_xyzs_raw + t_vec, labels_pred_rgb)


def process_laz(fpath, pnet):
    laz_data = LAZElem(fpath)
    prefix = fpath.split(os.sep)[-1].replace('.laz', '')
    out_dir = fpath + '-outs'
    laz_data.load()
    nx, ny = laz_data.num_xy_tiles
    tile_centers = np.rollaxis(np.array(np.meshgrid(np.arange(nx), np.arange(ny))), 0, 3).reshape([-1, 2])
    utils_.force_makedir(out_dir)
    ex = ThreadPoolExecutor(max_workers=cpu_count())
    for tc in tile_centers:
        ex.submit(clf_pcl, tc, pnet, out_dir, prefix, laz_data)
    ex.shutdown()


if __name__ == '__main__':

    # ret = laz_data.sample_tile([22, 1])
    # tile_xyzs_raw, t_vec, _ = ret
    # tile_xyzs = utils_.xyz_preprocess(tile_xyzs_raw)
    #
    # if tile_xyzs.shape[0] > 15000:
    #     idx = np.random.choice(np.arange(tile_xyzs.shape[0]), 15000)
    # tile_xyzs = tile_xyzs[idx]
    # scores = pnet.infer(tile_xyzs)
    # labels = utils_.scores2labels(scores, tile_xyzs)
    # fp = 'clf.ply'
    # labels_pred_rgb = utils_.new_colors[labels]
    # utils_.write_ply(fp, tile_xyzs, labels_pred_rgb)

    pnet = PointNet(mode='infer')
    pnet.load_weights('aerial-pointnet-weights.19-1.24.hdf5')
    laz_fpaths = glob(utils_.DIR + os.sep + '*.laz')
    for laz_fp in laz_fpaths:
        process_laz(laz_fp, pnet)
