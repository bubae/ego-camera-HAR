import os
import os.path as osp
import numpy as np
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C


__C.OPT_FLOW = edict();

__C.OPT_FLOW.SURF_THRESHOLD = 400

__C.OPT_FLOW.KEYPOINT_THRESHOLD = 20;

__C.TRAIN = edict();

__C.TRAIN.SCALES = (224,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 224

__C.TRAIN.NUM_STACK = 10;

__C.TRAIN.IMS_PER_BATCH = 16

__C.TRAIN.BATCH_SIZE = 16

# Iterations between snapshots
__C.TRAIN.SNAPSHOT_ITERS = 4000

# __C.TRAIN.PIXEL_MEANS = np.array([[[235.72685743, 238.83191218, 240.86580604]]])

__C.TRAIN.PIXEL_MEANS = np.array([127.34931653, 127.34931653, 127.34931653])

__C.TRAIN.FLOWNET = True

# 45.5764717,108.26965068,140.39894728
# 61.5835248    88.80047228  103.82588699 for ADL dataset
__C.TRAIN.SNAPSHOT_INFIX = ''

__C.TEST = edict();

__C.TEST.NUM_STACK = 10;

__C.TEST.PIXEL_MEANS = np.array([127.34931653, 127.34931653, 127.34931653])

__C.LSTM = edict();

__C.LSTM.NUM_STACK = 10;