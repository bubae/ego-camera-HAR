import init_path
from easydict import EasyDict as edict
from init_path import cfg
from Db import Db
import numpy as np 
import os, sys
import tensorflow as tf
tf.python.control_flow_ops = tf
import keras.backend.tensorflow_backend as KTF
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, merge, LSTM, Dense, Activation, Dropout
from keras.callbacks import History 
from keras.optimizers import SGD
from sklearn.decomposition import PCA
from sklearn import svm
# from sklearn.model_selection import train_test_split
from sendMail import send_mail
from motion_net.config import cfg as net_cfg
import glob


def get_session(gpu_fraction=0.90):
	'''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



def dataSetup(db, xid, timesteps):
	cnt = 0;

	objFeatureDir = "/home/bbu/Workspace/data/result/%s/Objectfeature/S_%d/fc7" % (db.name, testid);
	mtnFeatureDir = "/home/bbu/Workspace/data/result/%s/Motionfeature/S_%d_10/fc7" % (db.name, testid);




	imgList = glob.glob("/home/bbu/Workspace/data/result/%s/Motionfeature/S_%d_10/fc7/*_flip.npy" % (db.name, testid));

	iids = np.array([int(os.path.basename(path)[:-9]) for path in imgList]);
	iids = np.sort(iids);

	test_iids = np.array([iid for iid in iids if len(np.where(xid == db.iid2xid[iid])[0]) > 0])

	# print train_iids.shape, test_iids.shape

	numData = len(test_iids);

	iids = test_iids;

	# iids = np.where(db.iid2xid == xid)[0];

	# numData = len(iids) - timesteps;

	obj_test = [];
	mtn_test = [];

	for i in xrange(numData):
		processed_ims = [];
		iid = iids[i];

		for j in xrange(iid, iid+timesteps):
			objfeature = np.load(os.path.join(objFeatureDir, "%d.npy" % j));
			processed_ims.append(objfeature);
		obj_test.append(np.reshape(processed_ims, (2048)));
		# obj_test.append(np.reshape(processed_ims, (timesteps,2048)));

		processed_ims = [];
		for j in xrange(iid, iid+timesteps):
			mtnfeature = np.load(os.path.join(mtnFeatureDir, "%d.npy" % j));
			processed_ims.append(mtnfeature);
		mtn_test.append(np.reshape(processed_ims, (4096)));
		# mtn_test.append(np.reshape(processed_ims, (timesteps,4096)));

	obj_test = np.array(obj_test);
	mtn_test = np.array(mtn_test);

	# print obj_test.shape, mtn_test.shape
	return obj_test, mtn_test

def LSTM_Test(db, test_id, timesteps=5):

	modelPath = '/home/bbu/Workspace/data/result/GTEA/models/fully/multi_task/dropout/active_%d_10_1_fc7_dropout_model_config.json' % (testid);
	weightPath = '/home/bbu/Workspace/data/result/GTEA/models/fully/multi_task/dropout/active_%d_10_1_fc7_dropout_model_weight.h5' % (testid);

	# modelPath = '/home/bbu/Workspace/data/result/GTEA/models/lstm/multi_task/dropout/active_%d_1_%d_fc7_dropout_model_config.json' % (testid, timesteps);
	# weightPath = '/home/bbu/Workspace/data/result/GTEA/models/lstm/multi_task/dropout/active_%d_1_%d_fc7_dropout_model_weight.h5' % (testid, timesteps);


	# modelPath = '/home/bbu/Workspace/data/result/GTEA/models/lstm/multi_task/dropout/active_%d_1_%d_dropout_model_config.json' % (testid, timesteps);
	# weightPath = '/home/bbu/Workspace/data/result/GTEA/models/lstm/multi_task/dropout/active_%d_1_%d_dropout_model_weight.h5' % (testid, timesteps);
	json_data=open(modelPath).read()

	model = model_from_json(json_data)
	model.load_weights(weightPath)

	print "Model Loaded";

	cnt = 0;
	long_cnt = 0;
	short_cnt = 0;
	accuracy_score = 0;
	long_score = 0;
	short_score = 0;

	test_xids = np.where(db.xid2sid == test_id)[0];

	for xid in test_xids:
		label = db.xid2acid[xid];

		obj_test, mtn_test = dataSetup(db, xid, timesteps);

		if obj_test.shape[0] ==0:
			continue;

		# print obj_test.shape, mtn_test.shape
		classes = model.predict({'object_input':obj_test, 'motion_input': mtn_test}, batch_size=1)

		obj_predict = classes[0];
		mtn_predict = classes[1];
		act_predict = classes[2];

		# print act_predict.shape

		act_predict_sum = act_predict.sum(axis=0);

		# print act_predict_sum;

		predict = np.argmax(act_predict_sum);

		# print label, predict
		# act_label = db.xid2acid[xid];


		if act_predict.shape[0] > 30:
			long_cnt +=1;
			if int(label) == int(predict):
				long_score +=1;

		else:
			short_cnt +=1;
			if int(label) == int(predict):
				short_score +=1;

		if int(label) == int(predict):
			accuracy_score +=1;


		cnt = cnt +1;

		# print "Task %d : %d / %d" % (xid, cnt, len(test_xids))

	accuracy = np.float32(accuracy_score) / np.float32(len(test_xids))
	long_accuracy = np.float32(long_score) / np.float32(long_cnt)
	short_accuracy = np.float(short_score) / np.float(short_cnt);

	print "\nTest ID: %d" % test_id
	print accuracy, long_accuracy, short_accuracy, accuracy_score, cnt, len(test_xids);
	return accuracy
	# obj_train, mtn_train, obj_test, mtn_test, y_train_act, y_test_act, y_train_obj, y_test_obj, y_train_mtn, y_test_mtn, test_labels_act, test_labels_obj, test_labels_mtn = dataSetup(db, timesteps, testid);
	# print "Loaded Data"


	# classes = model.predict({'object_input':obj_test, 'motion_input': mtn_test}, batch_size=32)

	# obj_predict = np.argmax(classes[0], axis=1)
	# mtn_predict = np.argmax(classes[1], axis=1)
	# act_predict = np.argmax(classes[2], axis=1)

	# print np.array(classes[0]).shape
	# print np.array(classes[1]).shape
	# print np.array(classes[:,0]).shape


if __name__ == "__main__":
	setting = edict();
	setting.gpu 			= 1;
	setting.db 				= cfg.db.gtea;

	db = Db(setting.db, cfg.dstDir);
	db.genDb();

	# print np.where(np.where(db.iid2xid == 0)[0])

	# KTF.set_session(get_session())

	timesteps = 1;

	for testid in [3]:
		LSTM_Test(db, testid, timesteps);
