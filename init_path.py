# Authored by BUBae (widianpear@kaist.ac.kr)
# initialize paths

import os.path as osp
import sys
from easydict import EasyDict as edict

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

class_path = osp.join(this_dir, 'class');
add_path(class_path);

func_path = osp.join(this_dir, 'func');
add_path(func_path);

utils_path = osp.join(this_dir, 'utils');
add_path(utils_path);

data_path = osp.join('/home/bbu/Workspace/data');
add_path(data_path);

pycharm_debug_path = osp.join('/home/bbu/anaconda/envs/egoHAR/debug-eggs/pycharm-debug.egg');
add_path(pycharm_debug_path);

# idt_path = osp.join(this_dir, 'lib', 'improved_trajectory');
# add_path(idt_path);

# df_path = osp.join(this_dir, 'lib', 'dense_flow');
# add_path(df_path);

# Add caffe to PYTHONPATH
# caffe_path = osp.join(this_dir, 'lib', 'faster-rcnn', 'caffe-fast-rcnn', 'python')
caffe_path = osp.join(this_dir, 'caffe', 'python')
add_path(caffe_path)

# # Add lib to PYTHONPATH
# lib_path = osp.join(this_dir, 'lib', 'faster-rcnn', 'lib')
# add_path(lib_path)

#Add flownet path
# flownet_path = osp.join(this_dir, 'tools', 'flownet', 'models', 'FlowNetS');
# add_path(flownet_path)

dataset_path = osp.join(this_dir, 'func', 'datasets');
add_path(dataset_path);

__C = edict();

cfg = __C;

cfg.email 					= 'widianpear@kaist.ac.kr';

cfg.lib = edict();

cfg.db = edict();

cfg.db.gtea = edict();
cfg.db.gtea.name 			= 'GTEA';
cfg.db.gtea.funh 			= 'DB_GTEA';
cfg.db.gtea.root 			= osp.join(data_path, cfg.db.gtea.name);
cfg.db.gtea.frameRate 		= 1;

cfg.db.gtea_gaze = edict();
cfg.db.gtea_gaze.name 		= 'GTEA_GAZE';
cfg.db.gtea_gaze.funh 		= 'DB_GTEA_GAZE';
cfg.db.gtea_gaze.root 			= osp.join(data_path, cfg.db.gtea_gaze.name);
cfg.db.gtea.frameRate 		= 1;

cfg.db.gtea_gaze_plus = edict();
cfg.db.gtea_gaze_plus.name 		= 'GTEA_GAZE_PLUS';
cfg.db.gtea_gaze_plus.funh 		= 'DB_GTEA_GAZE_PLUS';
cfg.db.gtea_gaze_plus.root 			= osp.join(data_path, cfg.db.gtea_gaze_plus.name);
cfg.db.gtea.frameRate 		= 1;

cfg.db.kitchen = edict();
cfg.db.kitchen.name 		= 'KITCHEN';
cfg.db.kitchen.funh 		= 'DB_KITCHEN';
cfg.db.kitchen.root 		= osp.join(data_path, cfg.db.kitchen.name);
cfg.db.kitchen.frameRate 	= 1;


# cfg.db.gtea_flow = edict();
# cfg.db.gtea_flow.name 			= 'GTEA_FLOW';
# cfg.db.gtea_flow.funh 			= 'DB_GTEA_FLOW';
# cfg.db.gtea_flow.root 			= osp.join(data_path, cfg.db.gtea.name);
# cfg.db.gtea_flow.frameRate 		= 1;

cfg.db.adl = edict();
cfg.db.adl.name 			= 'ADL';
cfg.db.adl.funh 			= 'DB_ADL';
cfg.db.adl.root 			= osp.join(data_path, cfg.db.adl.name);
cfg.db.adl.frameRate 		= 2;

cfg.MODELS_DIR 				= osp.join(data_path, 'models');
cfg.dstDir 					= osp.join(data_path, 'result');

cfg.TRAIN = edict();
cfg.TEST = edict();

