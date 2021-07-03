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
from pointnet.model import PointNet


if __name__ == '__main__':
    ply_fps = glob(utils_.GT_DIR + os.sep + '*rgb.ply')
    pnet = PointNet(mode='train')
    pnet.load_data(ply_fps[:-1], [ply_fps[-1]])
    pnet.load_weights(utils_.DIR + '/trained_models/aerial-pointnet-weights.19-1.24.hdf5')
    pnet.train()
