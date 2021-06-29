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

label_colors = [(0, 0, 0), (244, 35, 231), (152, 250, 152),
                # 0 = unlabelled, 1 = man-made-terrain, 2 = natural-terrain
                (106, 142, 35), (190, 153, 153), (69, 69, 69),
                # 3 = high-vegetation, 4 = low-vegetation, 5 = buildings,
                (128, 64, 128), (255, 255, 255), (0, 0, 142)]
                # 6 = hard-scape, 7 = scan-artifacts, 8 = cars
label_colors = np.array(label_colors)
label_names = ['unlabelled', 'man-made-terrain', 'natural-terrain', 'high-vegetation', 'low-vegetation',
               'buildings', 'hard-scape', 'scan-artifacts', 'cars']


def z2rgb(zs_, min_z=-600, max_z=120):
    zs = np.clip(zs_, min_z, max_z)
    m = zs - zs.min()
    m = m / m.max()
    m = np.tile(np.expand_dims(m, -1), [1, 3])
    rgbs = (m * [0, 255, 0] + (1 - m) * [0, 0, 255]).astype(np.uint8)


def normalize_xyzs(xyzs_):
    xyzs = xyzs_ - xyzs_.min(axis=0)
    return xyzs / xyzs.max()
