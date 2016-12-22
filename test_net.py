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
# from motion_net.test import test_net, extract_motion_feature
from object_net.test import test_net, extract_object_feature, get_y_predict
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2


def modelPredictLabel(db):
	# history = History()
	hist_list = [];


	# dataSetup_length(db);

	for testid in [0,1,2,3]:
		data_dim = 4096
		timesteps = 10

		obj_train, mtn_train, obj_test, mtn_test, y_train_act, y_test_act, y_train_obj, y_test_obj, y_train_mtn, y_test_mtn, test_labels_act, test_labels_obj, test_labels_mtn = dataSetup(db, timesteps, testid);
		print "Loaded Data"

		modelPath = '/home/bbu/Workspace/data/result/GTEA/models/lstm/multi_task/active_%d_1_model_config.json' % testid;
		weightPath = '/home/bbu/Workspace/data/result/GTEA/models/lstm/multi_task/active_%d_1_model_weight.h5' % testid;
		json_data=open(modelPath).read()

		model = model_from_json(json_data)
		model.load_weights(weightPath)

		classes = model.predict({'object_input':obj_test, 'motion_input': mtn_test}, batch_size=32)

		obj_predict = np.argmax(classes[0], axis=1)
		mtn_predict = np.argmax(classes[1], axis=1)
		act_predict = np.argmax(classes[2], axis=1)

		print np.array(classes[0]).shape
		print np.array(classes[1]).shape
		print np.array(classes[:,0]).shape
		np.save('data/result/confusion_matrix/predict_result_%d.npy' % testid, np.array([test_labels_obj, test_labels_mtn, test_labels_act, obj_predict, mtn_predict, act_predict]))

		np.savetxt('data/result/confusion_matrix/predict_result_%d.npy' % testid, np.array([test_labels_obj, test_labels_mtn, test_labels_act, obj_predict, mtn_predict, act_predict]), delimiter=',')


		print y_train_act.shape, y_test_act.shape, y_train_obj.shape, y_test_obj.shape, y_train_mtn.shape, y_test_mtn.shape

if __name__ == "__main__":
	setting = edict();
	setting.gpu 			= 1;
	setting.db 				= cfg.db.gtea;

	db = Db(setting.db, cfg.dstDir);
	db.genDb();

	caffe.set_mode_gpu();
	caffe.set_device(0);

	modelPredictLabel(db);

	# accuracy_list = [];


	# ObjectNet Test Part
	# for i in [0,1,2,3]:
	# 	prototxtPath = 'models/%s/objectnet/VGG_CNN_M_2048/test.prototxt' % db.name;
	# 	# modelPath = '/home/bbu/Workspace/data/result/%s/snapshot/object_net/vgg_cnn_m_2048_centercropimage_iter_%d_%d.caffemodel' % (db.name, i, 16000);

	# 	modelPath = '/home/bbu/Workspace/data/result/%s/snapshot/object_net/vgg_cnn_m_2048_cropimage_iter_%d_20000.caffemodel' % (db.name, i);
	# 	accuracy = test_net(prototxtPath, modelPath, i, db);
	# 	accuracy_list.append(accuracy);

	## ObjectNet Feature Extract Part
	# for i in [0,1,2,3]:
	# 	prototxtPath = 'models/%s/objectnet/VGG_CNN_M_2048/deploy.prototxt' % db.name;
	# 	# modelPath = '/home/bbu/Workspace/data/result/%s/snapshot/object_net/vgg_cnn_m_2048_centercropimage_iter_%d_%d.caffemodel' % (db.name, i, 16000);

	# 	modelPath = '/home/bbu/Workspace/data/result/%s/snapshot/object_net/vgg_cnn_m_2048_cropimage_iter_%d_20000.caffemodel' % (db.name, i);

	# 	extract_object_feature(prototxtPath, modelPath, db, i);
		# accuracy = test_net(prototxtPath, modelPath, i, db);
		# accuracy_list.append(accuracy);

	# ObjectNet Predict Result
	# for i in [0,1,2,3]:
	# 	prototxtPath = 'models/%s/objectnet/VGG_CNN_M_2048/deploy.prototxt' % db.name;
	# 	modelPath = '/home/bbu/Workspace/data/result/%s/snapshot/object_net/vgg_cnn_m_2048_cropimage_iter_%d_20000.caffemodel' % (db.name, i);

	# 	get_y_predict(prototxtPath, modelPath, db, i);
		

	# testid = 0;

	# prototxtPath = 'models/%s/motionnet/test.prototxt' % db.name;

	# modelPath = '/home/bbu/Workspace/data/result/%s/snapshot/motion_net/motionnet_train_iter_%d_%d.caffemodel' % (db.name, testid, 20000);

	# extract_motion_feature(prototxtPath, modelPath, db, testid);

	## Motion Feature Extract Part
	# for i in [0,1,2,3]:
	# 	prototxtPath = 'models/%s/motionnet/ONE_IMAGE/test.prototxt' % db.name;

	# 	modelPath = '/home/bbu/Workspace/data/result/%s/snapshot/motion_net/motionnet_train_oneimage_backgroundon_iter_%d_%d.caffemodel' % (db.name, i, 20000);

	# 	accuracy = test_net(prototxtPath, modelPath, i, db);
	# 	accuracy_list.append(accuracy);
	# 	extract_motion_feature(prototxtPath, modelPath, db, i);

		# modelPath = '/home/bbu/Workspace/data/result/GTEA/motionnet_train__iter_10000.caffemodel';		
		# modelPath = '/home/bbu/Workspace/data/result/%s/snapshot/object_net/vgg_cnn_m_2048_iter_%d_5000.caffemodel' % (db.name, i);
		# accuracy = test_net(prototxtPath, modelPath, i, db);
		# accuracy_list.append(accuracy);
		# send_mail("widnanpear@gmail.com", "widianpear@kaist.ac.kr", [], "TEST Finished %d" % i, "TEST Finished %d" % i, None);		


	# np.savetxt('data/result/motionNet/motion_test_oneimage_backgroundon.txt', accuracy_list, delimiter=',')

	# print accuracy_list

	# send_mail("widnanpear@gmail.com", "widianpear@kaist.ac.kr", [], "TEST Finished %d" % i, "TEST Finished %d" % i, None);		
