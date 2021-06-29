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


class PointStreamer:

    def __init__(self, pcl_fpath=None):
        if pcl_fpath is not None:
            self.pcl_fpath = pcl_fpath
            self.f = open(pcl_fpath, 'r')
        self.point_idx = 0
        self.num_epochs = 0

    def get_points(self, sz=None, fmt='ply'):
        l = self.f.readline().strip().split(' ')
        if not l:
            self.num_epochs += 1
            self.f.close()
            self.f = open(self.pcl_fpath, 'r')
            l = self.f.readline().strip().split(' ')
        t = 1
        data = [(float(l[0]), float(l[1]), float(l[2]), int(l[-3]), int(l[-2]), int(l[-1]))]
        while True:
            l = self.f.readline().strip().split(' ')
            if sz is not None:
                if t < sz:
                    if not l:
                        self.num_epochs += 1
                        self.f.close()
                        self.f = open(self.pcl_fpath, 'r')
                        l = self.f.readline().strip().split(' ')
            elif not sz:
                if not l:
                    break
            data.append((float(l[0]), float(l[1]), float(l[2]), int(l[-3]), int(l[-2]), int(l[-1])))
            # if t % 1000000 == 0:
            #     if fmt == 'ply':
            #         v = np.array(data,
            #                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'),
            #                             ('blue', 'u1')])
            #         el = PlyElement.describe(v, 'vertex')
            #         PlyData([el]).write('scratchspace/chunk-' + str(t) + '.ply')
            t += 1
            if t >= sz:
                break
        if fmt == 'ply':
            v_ = np.array(data,
                         dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
            v = PlyElement.describe(v_, 'vertex')
            # PlyData([v]).write('chunk-final.ply')
        return v

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
