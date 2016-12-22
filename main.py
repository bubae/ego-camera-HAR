import init_path
import numpy as np
import os, sys
from Db import Db
from init_path import cfg
from easydict import EasyDict as edict
from sendMail import send_mail
from timer import Timer
import cv2
import motion_net.calOF as calOF
# from motion_net.train import train_net
from motion_net.test import test_net, extract_motion_feature, get_y_predict
from object_net.train import train_net
from object_net.test import test_net
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
# from init_path import cfg
# from easydict import EasyDict as edict
# from sendMail import send_mail
# from timer import Timer
# import cv2
# from subprocess import call, check_output
# from utils.blob import im_list_to_blob, prep_im_for_blob
# from motion_net.config import cfg as net_cfg
from calMeanIMG import calMeanIMG, calMeanIMGFlow
# from object_net.test import extract_object_feature
from hand_segment.crop_image import cropROIImages

if __name__ == "__main__":
	setting = edict();
	setting.gpu 			= 1;
	setting.db 				= cfg.db.gtea;

	db = Db(setting.db, cfg.dstDir);
	db.genDb();

	caffe.set_mode_gpu();
	caffe.set_device(0);

	# calMeanIMG(db);

	# cropROIImages(db, setting);

	# calOF.motion_net(db, setting)

	# solver_prototxt = 'models/%s/objectnet/VGG_CNN_M_2048/solver.prototxt' % db.name;
	# pretrained_model = 'data/object_net/VGG_CNN_M_2048.caffemodel';

	# for test_id in [0,1,2,3]:
	# 	train_net(solver_prototxt, db, test_id, pretrained_model, 20000, 80);
	# 	send_mail("widnanpear@gmail.com", "widianpear@kaist.ac.kr", [], "Train Finished %d" % test_id, "Train Finished %d" % test_id, None);

	## ONE IMAGE MOTION NET TRAINING
	# solver_prototxt = 'models/%s/motionnet/ONE_IMAGE/solver.prototxt' % db.name;
	# pretrained_model = 'data/motion_net/vgg_16_action_flow_pretrain.caffemodel';

	# startIters = [0, 8000, 0, 4000];
	# for i in [2, 1, 3]:
	# 	startIter = startIters[i];		
	# 	if i==3:
	# 		pretrained_model = '/home/bbu/Workspace/data/result/%s/snapshot/motion_net/motionnet_train_oneimage_backgroundon_iter_%d_%d.caffemodel' % (db.name, i, startIter);
	# 	train_net(solver_prototxt, db, i, pretrained_model, 20000, startIter);


	## FIVE IMAGE MOTION NET TRAINING
	# solver_prototxt = 'models/%s/motionnet/FIVE_IMAGE/solver.prototxt' % db.name;
	# pretrained_model = 'data/motion_net/vgg_16_action_flow_pretrain.caffemodel';

	# startIters = [0, 0, 0, 0];
	# for i in [3]:
	# 	startIter = startIters[i];
	# 	train_net(solver_prototxt, db, i, pretrained_model, 20000, startIter);


	## TEN IMAGE MOTION NET TRAINING
	# solver_prototxt = 'models/%s/motionnet/solver.prototxt' % db.name;
	# pretrained_model = 'data/motion_net/vgg_16_action_flow_pretrain.caffemodel';

	# startIters = [0, 0, 0, 0];
	# for i in [0,1,2,3]:
	# 	startIter = startIters[i];
	# 	train_net(solver_prototxt, db, i, pretrained_model, 20000, startIter);



	# for i in [80, 100, 140, 160, 180]:

	# 	# train_net(solver_prototxt, db, 1, pretrained_model, 20000, i);

	# 	prototxtPath = 'models/%s/objectnet/VGG_CNN_M_2048/test.prototxt' % db.name;
	# 	modelPath = '/home/bbu/Workspace/data/result/%s/snapshot/object_net/r_%d/vgg_cnn_m_2048_cropimage_iter_1_20000.caffemodel' % (db.name, i);
	# 	test_net(prototxtPath, modelPath, 1, db, i);
	# 	send_mail("widnanpear@gmail.com", "widianpear@kaist.ac.kr", [], "Test Finished %d" % i, "Test Finished %d" % i, None);


	# solver_prototxt = 'models/%s/motionnet/ONE_IMAGE/solver.prototxt' % db.name;
	# pretrained_model = 'data/motion_net/vgg_16_action_flow_pretrain.caffemodel';

	# startIters = [0, 0, 0, 0, 0];
	# testset_id = 0;
	for i in [1]:
		# startIter = startIters[i]
		# if i==0:
		# 	pretrained_model = '/home/bbu/Workspace/data/result/%s/snapshot/motion_net/motionnet_train_oneimage_iter_%d_%d.caffemodel' % (db.name, i, startIter);
		# else:
		# 	pretrained_model = 'data/motion_net/vgg_16_action_flow_pretrain.caffemodel';
			
		# train_net(solver_prototxt, db, i, pretrained_model, 16000, startIter);
		# else:
		# 	pretrained_model = 'data/motion_net/vgg_16_action_flow_pretrain.caffemodel';
		# 	train_net(solver_prototxt, db, i, pretrained_model, 30000, startIter);


		pretrained_model = '/home/bbu/Workspace/data/result/%s/snapshot/motion_net/motionnet_train_iter_%d_%d.caffemodel' % (db.name, i, 20000);
		prototxt_path = 'models/GTEA/motionnet/test.prototxt';

		get_y_predict(prototxt_path, pretrained_model, db, i);

		send_mail("widnanpear@gmail.com", "widianpear@kaist.ac.kr", [], "Train Finished %d" % i, "Train Finished %d" % i, None);


	# prototxtPath = 'models/%s/motionnet/ONE_IMAGE/test.prototxt' % db.name;

	# modelPath = '/home/bbu/Workspace/data/result/%s/snapshot/motion_net/motionnet_train_iter_%d_%d.caffemodel' % (db.name, 4, 30000);

	# extract_motion_feature(prototxtPath, modelPath, db);

	# send_mail("widnanpear@gmail.com", "widianpear@kaist.ac.kr", [], "Feafure Finished", "Feature Finished", None);