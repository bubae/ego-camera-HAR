import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.CROP = edict();

__C.CROP.ORIGIN_SIZE = (720, 404);

__C.CROP.NETWORK_SIZE = (360, 203);

__C.CROP.PIXEL_MEANS = np.array([[[45.5764717,108.26965068,140.39894728]]]);

__C.CROP.ALPHA = 6.4

__C.CROP.BETA = 0.2

__C.TRAIN = edict();

__C.TRAIN.SCALES = (203,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 360

__C.TRAIN.IMS_PER_BATCH = 8

__C.TRAIN.BATCH_SIZE = 8

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 4000

__C.TRAIN.PIXEL_MEANS = np.array([[[45.5764717,108.26965068,140.39894728]]])

# __C.TRAIN.FLOWNET = True

# 45.5764717,108.26965068,140.39894728
# 61.5835248    88.80047228  103.82588699 for ADL dataset
__C.TRAIN.SNAPSHOT_INFIX = ''

__C.TEST = edict();

# __C.TEST.PIXEL_MEANS = np.array([[[235.72685743, 238.83191218, 240.86580604]]])