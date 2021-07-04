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
import threading

import numpy as np
# import numpy_indexed as npi
from plyfile import PlyElement, PlyData

import utils_


class PlySet:

    def __init__(self, ply_fps):
        self.ply_fps = ply_fps
        self.curr_ply_idx = 0
        self.n_plys = len(ply_fps)
        self.ply_fnames = [fp.split(os.sep)[-1] for fp in self.ply_fps]
        self.ply_elems = [PlyElem(fp) for fp in self.ply_fps]
        self.val_set_idx = self.n_plys - 1
        # self.scales_matched = False
        # self.match_scales()

    # def match_scales(self, targ_idx=1, dump=False):
    #     if not self.scales_matched:
    #         for i in range(len(self.ply_elems)):
    #             if i == targ_idx:
    #                 continue
    #             self.ply_elems[i].rescale(utils_.scale_map[self.ply_elems[i].fname.replace('.ply', '.txt')])
    #             if dump:
    #                 out_ply_fpath = self.ply_elems[i].fpath.replace('.ply', '-rescaled.ply')
    #                 if not os.path.exists(out_ply_fpath):
    #                     self.ply_elems[i].dump(out_ply_fpath)
    #                 else:
    #                     print(out_ply_fpath, 'exists, skipping....')
    #         self.scales_matched = True

    def sample_tile(self):
        tile_data = self.ply_elems[self.curr_ply_idx].sample_tile()
        self.curr_ply_idx = (self.curr_ply_idx + 1) % self.n_plys
        return tile_data


class PlyElem:

    def __init__(self, ply_fpath):
        self.fpath = ply_fpath
        self.fname = ply_fpath.split(os.sep)[-1]
        self.loaded = False
        self.xyzs = None
        self.xyz_min = None
        self.xyz_max = None
        self.xs = None
        self.ys = None

        self.rgbs = None
        self.data = utils_.read_ply(self.fpath, raw=True)
        self.point_class_color_hashes = None
        self.rescale(utils_.scale_map[self.fname.replace('.ply', '.txt')])

    def sample_tile(self):
        self.load()
        while True:
            start_x = np.random.uniform(self.xyz_min[0], self.xyz_max[0] - utils_.POINT_TILER_SIDE)
            end_x = start_x + utils_.POINT_TILER_SIDE
            start_y = np.random.uniform(self.xyz_min[1], self.xyz_max[1] - utils_.POINT_TILER_SIDE)
            end_y = start_y + utils_.POINT_TILER_SIDE
            filt = np.logical_and(np.logical_and(np.logical_and(self.xs >= start_x, self.xs < end_x),
                                                 self.ys >= start_y), self.ys < end_y)
            xyzs = self.xyzs[filt]
            rgbs = self.rgbs[filt]
            color_hashes = utils_.color2hash(rgbs)
            filt = np.logical_and(np.logical_and(color_hashes != utils_.excluded_label_hashes[0],
                                                 color_hashes != utils_.excluded_label_hashes[1]),
                                  color_hashes != utils_.excluded_label_hashes[2])
            if filt[filt].shape[0] < 100:
                continue
            xyzs = xyzs[filt]
            rgbs = rgbs[filt]
            # xyzs = np.floor(xyzs / utils_.GRID_RESOLUTION)
            xyzs = xyzs - xyzs.min(axis=0)
            break
        idx = np.random.choice(np.arange(xyzs.shape[0]), utils_.MAX_TRAIN_IN_POINTS)
        # s = int(np.ceil(xyzs.shape[0] / utils_.MAX_TRAIN_IN_POINTS))
        # idx = np.arange(0, xyzs.shape[0], s)
        xyzs = xyzs[idx]
        rgbs = rgbs[idx]
        return xyzs, rgbs

    def load(self):
        if not self.loaded:
            self.xyzs, self.rgbs = utils_.parse_plydata(self.data)
            self.xyz_min = self.xyzs.min(axis=0)
            self.xyz_max = self.xyzs.max(axis=0)
            self.xs, self.ys, _ = self.xyzs.T
            self.point_class_color_hashes = utils_.color2hash(self.rgbs)
            self.loaded = True

    def rescale(self, scale):
        self.load()
        self.xyzs = self.xyzs * scale
        self.xyz_min = self.xyzs.min(axis=0)
        self.xyz_max = self.xyzs.max(axis=0)
        self.xs, self.ys, _ = self.xyzs.T

    def dump(self, fpath):
        self.load()
        utils_.write_ply(fpath, self.xyzs, self.rgbs)


class LAZElem:

    def __init__(self, laz_fpath):
        self.fpath = laz_fpath
        self.fname = self.fpath.split(os.sep)[-1]
        self.loaded = False
        self.xyzs = None
        self.xyz_max = None
        self.xs = None
        self.ys = None

        self.num_xy_tiles = None
        self.start_tile_x = 0
        self.start_tile_y = 0
        # self.t_vec = None

    def load(self):
        if not self.loaded:
            self.xyzs = utils_.read_laz(self.fpath)
            # self.xyzs = np.floor(utils_.normalize_xyzs(self.xyzs * utils_.LAZ_SCALE_CONST,
            #                                            scale_const=1. / utils_.GRID_RESOLUTION))
            self.xyzs = utils_.normalize_xyzs(self.xyzs, scale_const=utils_.LAZ_SCALE_CONST)
            # self.xyzs = npi.unique(self.xyzs)
            self.xs, self.ys, _ = self.xyzs.T
            self.xyz_max = self.xyzs.max(axis=0)
            self.num_xy_tiles = np.ceil(self.xyz_max[:2] / [utils_.POINT_TILER_SIDE,
                                                            utils_.POINT_TILER_SIDE]).astype(np.int)
            self.loaded = True

    def sample_tile(self, start_xy=None):
        self.load()
        if start_xy is None:
            start_tile_x, start_tile_y = self.start_tile_x, self.start_tile_y
        else:
            start_tile_x, start_tile_y = start_xy

        if start_tile_x >= self.num_xy_tiles[0] or start_tile_y > self.num_xy_tiles[1]:
            print('(☞ﾟヮﾟ)☞ LAZ file exhausted! ☜(ﾟヮﾟ☜)')
            return None

        start_x = start_tile_x * utils_.POINT_TILER_SIDE
        end_x = start_x + utils_.POINT_TILER_SIDE
        start_y = start_tile_y * utils_.POINT_TILER_SIDE
        end_y = start_y + utils_.POINT_TILER_SIDE

        filt = np.logical_and(np.logical_and(np.logical_and(self.xs >= start_x, self.xs < end_x),
                                             self.ys >= start_y), self.ys < end_y)
        xyzs = self.xyzs[filt]
        t_vec = xyzs.min(axis=0)
        xyzs = xyzs - t_vec

        if start_xy is None:
            if (self.start_tile_x + 1) >= self.num_xy_tiles[0]:
                self.start_tile_y += 1
                self.start_tile_x = 0
            else:
                self.start_tile_x += 1
        return xyzs, t_vec, None


class PointStreamer:

    def __init__(self, pcl_fpath, label_fpath):
        self.translate_const = None
        self.scale_const = None
        self.all_gt_pcl_fpaths = [pcl_fpath]
        self.all_gt_label_fpaths = [label_fpath]
        self.num_fpaths = len(self.all_gt_pcl_fpaths)
        self.curr_datafile_idx = 0
        self.pcl_fpath = self.all_gt_pcl_fpaths[0]
        self.pcl_label_fpath = self.all_gt_label_fpaths[0]
        self.f = open(self.pcl_fpath, 'rb')
        self.f_label = open(self.pcl_label_fpath, 'rb')
        self.label_loaded = True
        self.point_idx = 0
        self.num_epochs = 0

    def get_points(self, sz=None, stride=utils_.GT_STRIDE, normalize_point_locs=True, scale=utils_.GT_SCALE):
        l = self.f.readline().decode('utf-8').strip().split(' ')
        l_label = self.f_label.readline().decode('utf-8').strip()
        file_ended = False
        if len(l) < 2 or not l:
            self.num_epochs += 1
            self.f.close()
            if self.label_loaded:
                self.f_label.close()
            print(self.all_gt_pcl_fpaths[self.curr_datafile_idx], 'consumed, moving to next file...')
            self.curr_datafile_idx += 1
            self.curr_datafile_idx = self.curr_datafile_idx % self.num_fpaths
            self.pcl_fpath = self.all_gt_pcl_fpaths[self.curr_datafile_idx]
            self.pcl_label_fpath = self.all_gt_label_fpaths[self.curr_datafile_idx]
            self.f = open(self.pcl_fpath, 'rb')
            self.f_label = open(self.pcl_label_fpath, 'rb')
            print('Next file ->', self.pcl_fpath)
            return None, None
        t = 1
        data = [(float(l[0]), float(l[1]), float(l[2]), int(l[-3]), int(l[-2]), int(l[-1]))]
        data_label = None
        if self.label_loaded:
            data_label = [int(l_label)]
        while True:
            l = self.f.readline().decode('utf-8').strip().split(' ')
            if self.label_loaded:
                l_label = self.f_label.readline().decode('utf-8').strip()
            if len(l) < 2:
                self.num_epochs += 1
                self.f.close()
                if self.label_loaded:
                    self.f_label.close()
                file_ended = True
                break
            # if len(l_label) > 0:
            data.append((float(l[0]), float(l[1]), float(l[2]), int(l[-3]), int(l[-2]), int(l[-1])))
            if self.label_loaded:
                data_label.append(int(l_label))
            t += 1
            if sz is not None and t >= sz:
                break
        data = np.array(data)
        if self.label_loaded:
            data_label = np.array(data_label)
        xyzs_ = data[:, :3]
        if self.translate_const is None:
            self.translate_const = xyzs_.min(axis=0)
            print('Translation const ->', self.translate_const)
        if self.scale_const is None:
            tmp = xyzs_ - self.translate_const
            self.scale_const = 1. / tmp.max()
            print(' |--++', self.pcl_fpath, 'Scaling const ->', self.scale_const, "++--| ")
            # return None, None
        if normalize_point_locs:
            xyzs = utils_.normalize_xyzs(xyzs_, scale_const=self.scale_const,
                                         translate_const=self.translate_const) * scale
            xyzs_unique = npi.unique(xyzs)
            i_ = npi.indices(xyzs, xyzs_unique)
            data[:, :3] = xyzs
            data = data[i_]
            if self.label_loaded:
                data_label = data_label[i_]
        if file_ended:
            print(self.all_gt_pcl_fpaths[self.curr_datafile_idx], 'consumed, moving to next file...')
            self.curr_datafile_idx += 1
            self.curr_datafile_idx = self.curr_datafile_idx % self.num_fpaths
            self.f = open(self.all_gt_pcl_fpaths[self.curr_datafile_idx], 'rb')
            self.f_label = open(self.all_gt_label_fpaths[self.curr_datafile_idx], 'rb')
            self.pcl_fpath = self.all_gt_pcl_fpaths[self.curr_datafile_idx]
            self.pcl_label_fpath = self.all_gt_label_fpaths[self.curr_datafile_idx]
            print('Next file ->', self.pcl_fpath)

        idx = np.arange(0, data.shape[0], stride)

        return data[idx], data_label[idx]


class PlyIO:

    def __init__(self, point_streamer):
        self.point_streamer = point_streamer
        self.out_ply_fpath = self.point_streamer.pcl_fpath.split(os.sep)[-1].replace('.txt', '.ply')
        self.v = None

    def write_ply_buffered(self, xyzs, rgbs=None):
        data = []
        for i in range(xyzs.shape[0]):
            if rgbs is not None:
                data.append((float(xyzs[i][0]), float(xyzs[i][1]), float(xyzs[i][2]),
                             int(rgbs[i][-3]), int(rgbs[i][-2]), int(rgbs[i][-1])))
            else:
                data.append((float(xyzs[i][0]), float(xyzs[i][1]), float(xyzs[i][2])))
        if rgbs is not None:
            v = np.array(data,
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
                                ('blue', 'u1')])
        else:
            v = np.array(data,
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        if self.v is None:
            self.v = [v]
        else:
            self.v.append(v)
        v_ = np.squeeze(np.hstack(self.v))
        el = PlyElement.describe(v_, 'vertex')
        # print('Writing points to', self.out_ply_fpath, ' 👉 Number of points =', v_.shape[0], end='\r')
        PlyData([el]).write(self.out_ply_fpath)

    def dump_ply_worker(self, chunk_size=100000):
        while True:
            data, labels = self.point_streamer.get_points(chunk_size, utils_.GT_STRIDE)
            if labels is None:
                break
            if data is None:
                self.out_ply_fpath = self.point_streamer.pcl_fpath.split(os.sep)[-1].replace('.txt', '.ply')
                self.v = None
                return
            xyzs, rgbs = utils_.blend_data(data, labels)
            self.write_ply_buffered(xyzs, rgbs)

    def dump_ply(self):
        # self.dump_ply_worker()
        th = threading.Thread(target=self.dump_ply_worker)
        th.start()
