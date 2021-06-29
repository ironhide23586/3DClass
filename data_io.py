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


import numpy as np
from tqdm import tqdm
from plyfile import PlyElement, PlyData
import pylas

import utils


class PointStreamer:

    def __init__(self, pcl_fpath=None, pcl_label_fpath=None):
        if pcl_fpath is not None:
            self.pcl_fpath = pcl_fpath
            self.f = open(pcl_fpath, 'rb')
        self.label_loaded = False
        if pcl_fpath is not None:
            self.pcl_label_fpath = pcl_label_fpath
            self.f_label = open(pcl_label_fpath, 'rb')
            self.label_loaded = True
        self.point_idx = 0
        self.num_epochs = 0

    def get_points(self, sz=None, stride=1, normalize_point_locs=False, scale=1.):
        l = self.f.readline().decode('utf-8').strip().split(' ')
        l_label = self.f_label.readline().decode('utf-8').strip()
        if len(l) < 2 or not l:
            self.num_epochs += 1
            self.f.close()
            self.f = open(self.pcl_fpath, 'rb')
            l = self.f.readline().decode('utf-8').strip().split(' ')
            if self.label_loaded:
                self.f_label.close()
                self.f_label = open(self.pcl_label_fpath, 'rb')
                l_label = self.f.readline().decode('utf-8').strip()
        t = 1
        data = [(float(l[0]), float(l[1]), float(l[2]), int(l[-3]), int(l[-2]), int(l[-1]))]
        ksize = self.f.tell()
        data_label = None
        if self.label_loaded:
            data_label = [int(l_label)]
            ksize_label = self.f_label.tell()
        while True:
            if t % stride != 0:
                t += stride - 1
                self.f.seek(stride * ksize, 1)
                self.f.readline()
                if self.label_loaded:
                    self.f_label.seek(stride * ksize_label, 1)
                    self.f_label.readline()
                continue
            l = self.f.readline().decode('utf-8').strip().split(' ')
            if self.label_loaded:
                l_label = self.f_label.readline().decode('utf-8').strip()
            if len(l) < 2:
                self.num_epochs += 1
                self.f.close()
                self.f = open(self.pcl_fpath, 'rb')
                if self.label_loaded:
                    self.f_label.close()
                    self.f_label = open(self.pcl_label_fpath, 'rb')
                break
            if len(l_label) > 0:
                data.append((float(l[0]), float(l[1]), float(l[2]), int(l[-3]), int(l[-2]), int(l[-1])))
                if self.label_loaded:
                    data_label.append(int(l_label))
            t += 1
            if sz is not None and (t // stride) >= sz:
                break
        data = np.array(data)
        if self.label_loaded:
            data_label = np.array(data_label)
        if normalize_point_locs:
            xyzs = utils.normalize_xyzs(data[:, :3]) * scale
            data[:, :3] = xyzs
        return data, data_label

    def write_ply(self, xyzs, rgbs=None, fpath='tmp.ply'):
        data = []
        print('Writing points to', fpath)
        for i in tqdm(range(xyzs.shape[0])):
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
        el = PlyElement.describe(v, 'vertex')
        PlyData([el]).write(fpath)

    def read_ply(self, fpath):
        print('ψ(｀∇´)ψ Reading', fpath)
        ply_data = PlyData.read(fpath)
        xs = ply_data['vertex']['x']
        ys = ply_data['vertex']['y']
        zs = ply_data['vertex']['z']
        xyzs = np.vstack([xs, ys, zs]).T
        rgbs = None
        print('⚡ Found', xs.shape[0], 'points')
        if 'red' in ply_data['vertex']:
            print('Found RGB data ☢')
            rs = ply_data['vertex']['red']
            gs = ply_data['vertex']['green']
            bs = ply_data['vertex']['blue']
            rgbs = np.vstack([rs, gs, bs]).T
        return xyzs, rgbs

    def read_laz(self, fpath):
        las = pylas.read(fpath)
        xyzs = np.vstack([las['X'], las['Y'], las['Z']]).T
        return xyzs