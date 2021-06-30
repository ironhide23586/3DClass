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
from glob import glob

import numpy as np
import numpy_indexed as npi
from plyfile import PlyElement, PlyData

import utils_


class PlySet:

    def __init__(self, ply_fps):
        self.ply_fps = ply_fps
        self.ply_data = [PlyElem(fp) for fp in self.ply_fps]

    def match_scales(self):
        k = 0


class PlyElem:

    def __init__(self, ply_fpath):
        self.ply_fpath = ply_fpath
        self.ply_data = utils_.read_ply(self.ply_fpath, raw=True)
        self.point_class_color_hashes = None
        self.xyzs = None
        self.rgbs = None
        self.ply_mmaped = False
        self.class_pointset_map = {}  # maps class color hashes to their respective point-sets
        self.loaded = False

    def load(self):
        if not self.loaded:
            self.xyzs, self.rgbs = utils_.parse_plydata(self.ply_data)
            self.point_class_color_hashes = utils_.color2hash(self.rgbs)
            self.loaded = True

    def rescale(self, scale):
        self.load()
        self.xyzs = self.xyzs * scale

    def populate_class_pointset_map(self):
        for color_hash in utils_.label_color_hashes:
            self.class_pointset_map[color_hash] = self.xyzs[self.point_class_color_hashes == color_hash]

    def dump(self, fpath):
        utils_.write_ply(self.xyzs, fpath, rgbs=self.rgbs)


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
            # print('Translation const ->', self.translate_const)
        if self.scale_const is None:
            tmp = xyzs_ - self.translate_const
            self.scale_const = 1. / tmp.max()
            print(' |--++', self.pcl_fpath, 'Scaling const ->', self.scale_const, "++--| ")
            return None, None
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
