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

def get_session(gpu_fraction=0.80):
	'''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

def dataSetup(db, timesteps, testid):

	objFeatureDir = "/home/bbu/Workspace/data/result/%s/Objectfeature/fc7" % db.name;
	mtnFeatureDir = "/home/bbu/Workspace/data/result/%s/Motionfeature/S_%d_1/fc7" % (db.name, testid);

	imgList = glob.glob("/home/bbu/Workspace/data/result/%s/Motionfeature/S_%d_1/fc7/*_flip.npy" % (db.name, testid));

	iids = np.array([int(os.path.basename(path)[:-9]) for path in imgList]);
	iids = np.sort(iids);
	labels = np.array([int(db.xid2acid[db.iid2xid[iid]]) for iid in iids])

	train_xids = np.where(db.xid2sid != testid)[0];

	train_iid_list = [np.where(db.iid2xid == xid)[0][:-(net_cfg.LSTM.NUM_STACK)] for xid in train_xids];

	train_iids = np.array([]);
	train_iids = train_iids.astype(int)

	for iid_list in train_iid_list:
		train_iids = np.concatenate((train_iids, iid_list));

	test_xids = np.where(db.xid2sid == testid)[0];

	test_iid_list = [np.where(db.iid2xid == xid)[0][:-(net_cfg.LSTM.NUM_STACK)] for xid in test_xids];

	test_iids = np.array([]);

	for iid_list in test_iid_list:
		test_iids = np.concatenate((test_iids, iid_list));

	test_iids = test_iids.astype(int)

	tids = np.array([db.iid2xid[iid] in train_xids for iid in iids])

	# print tids.shape, train_iids.shape, test_iids.shape
	# print labels.shape, train_iids[-1]
	train_labels = db.xid2acid[db.iid2xid[train_iids].astype(int)].astype(int)
	test_labels = db.xid2acid[db.iid2xid[test_iids].astype(int)].astype(int)
	# train_iids = iids[np.where(tids == True)[0]];
	# train_labels = labels[np.where(tids == True)[0]];
	# test_iids = iids[np.where(tids == False)[0]];
	# test_labels = labels[np.where(tids == False)[0]];

	obj_train = [];
	mtn_train = [];
	obj_test = [];
	mtn_test = [];

	for iid in train_iids:
		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			objfeature = np.load(os.path.join(objFeatureDir, "%d.npy" % iid));
			processed_ims.append(objfeature);
		# obj_train.append(np.reshape(processed_ims, (4096)));
		obj_train.append(np.reshape(processed_ims, (timesteps,2048)));

		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			mtnfeature = np.load(os.path.join(mtnFeatureDir, "%d.npy" % iid));
			processed_ims.append(mtnfeature);
		# mtn_train.append(np.reshape(processed_ims, (4096)));
		mtn_train.append(np.reshape(processed_ims, (timesteps,4096)));

	for iid in test_iids:
		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			objfeature = np.load(os.path.join(objFeatureDir, "%d.npy" % iid));
			processed_ims.append(objfeature);
		# obj_test.append(np.reshape(processed_ims, (4096)));
		obj_test.append(np.reshape(processed_ims, (timesteps,2048)));

		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			mtnfeature = np.load(os.path.join(mtnFeatureDir, "%d.npy" % iid));
			processed_ims.append(mtnfeature);
		# mtn_test.append(np.reshape(processed_ims, (4096)));
		mtn_test.append(np.reshape(processed_ims, (timesteps,4096)));


		# objfeature = np.load(os.path.join(objFeatureDir, "%d.npy" % iid));
		# mtnfeature = np.load(os.path.join(mtnFeatureDir, "%d.npy" % iid));
		# # obj_test.append(objfeature[0])
		# # mtn_test.append(mtnfeature)
		# obj_test.append(np.reshape(objfeature, (1,4096)));
		# mtn_test.append(np.reshape(mtnfeature, (1,4096)));

	obj_train = np.array(obj_train);
	mtn_train = np.array(mtn_train);
	obj_test = np.array(obj_test);
	mtn_test = np.array(mtn_test);

	print obj_train.shape, mtn_train.shape

	train_labels = db.xid2acid[db.iid2xid[train_iids].astype(int)].astype(int)
	test_labels_act = db.xid2acid[db.iid2xid[test_iids].astype(int)].astype(int)

	y_train_act = np.zeros((len(train_iids), max(db.xid2acid)+1))
	y_train_act[np.arange(len(train_iids)), train_labels] = 1
	y_test_act = np.zeros((len(test_iids), max(db.xid2acid)+1))
	y_test_act[np.arange(len(test_iids)), test_labels_act] = 1

	train_labels = db.xid2ocid[db.iid2xid[train_iids].astype(int)].astype(int)
	test_labels_obj = db.xid2ocid[db.iid2xid[test_iids].astype(int)].astype(int)

	y_train_obj = np.zeros((len(train_iids), max(db.xid2ocid)+1))
	y_train_obj[np.arange(len(train_iids)), train_labels] = 1
	y_test_obj = np.zeros((len(test_iids), max(db.xid2ocid)+1))
	y_test_obj[np.arange(len(test_iids)), test_labels_obj] = 1

	train_labels = db.xid2mcid[db.iid2xid[train_iids].astype(int)].astype(int)
	test_labels_mtn = db.xid2mcid[db.iid2xid[test_iids].astype(int)].astype(int)

	y_train_mtn = np.zeros((len(train_iids), max(db.xid2mcid)+1))
	y_train_mtn[np.arange(len(train_iids)), train_labels] = 1
	y_test_mtn = np.zeros((len(test_iids), max(db.xid2mcid)+1))
	y_test_mtn[np.arange(len(test_iids)), test_labels_mtn] = 1	



	return obj_train, mtn_train, obj_test, mtn_test, y_train_act, y_test_act, y_train_obj, y_test_obj, y_train_mtn, y_test_mtn, test_labels_act, test_labels_obj, test_labels_mtn
	# for i in xrange(len(iid2xid)):


def dataSetup_length(db, timestep, testid):
	## Accuracy comparison based on sequence length
	# instanceNum = len(db.xid2sid)

	test_xids = np.where(db.xid2sid == testid)[0];

	print test_xids
	long_xids = [];
	short_xids = [];

	cnt = 0;
	for xid in test_xids:
		if len(np.where(db.iid2xid == xid)[0]) > 30:
			long_xids.append(xid);
			cnt = cnt + 1;
		else:
			short_xids.append(xid);

	print long_xids, short_xids, cnt;

	# iids = np.array([int(os.path.basename(path)[:-9]) for path in imgList]);
	# iids = np.sort(iids);
	# labels = np.array([int(db.xid2acid[db.iid2xid[iid]]) for iid in iids])

	# train_iid_list = [np.where(db.iid2xid == xid)[0][:-(net_cfg.LSTM.NUM_STACK)] for xid in train_xids];

	# train_iids = np.array([]);
	# train_iids = train_iids.astype(int)

	# for iid_list in train_iid_list:
	# 	train_iids = np.concatenate((train_iids, iid_list));


	# return obj_test_long, mtn_test_long, obj_test_short, mtn_test_short, obj_test_long_label, mtn_test_long_label, act_test_long_label, obj_test_short_label, mtn_test_short_label, act_test_short_label 

def xid2InputData(db, testid, timesteps, xid):
	objFeatureDir = "/home/bbu/Workspace/data/result/%s/Objectfeature/fc6" % db.name;
	mtnFeatureDir = "/home/bbu/Workspace/data/result/%s/Motionfeature/S_%d_1/fc6" % (db.name, testid);

	imgList = glob.glob("/home/bbu/Workspace/data/result/%s/Motionfeature/S_%d_1/fc6/*_flip.npy" % (db.name, testid));

	# iids = np.array([int(os.path.basename(path)[:-9]) for path in imgList]);
	# iids = np.sort(iids);
	# labels = np.array([int(db.xid2acid[db.iid2xid[iid]]) for iid in iids])

	iid_list = np.where(db.iid2xid == xid)[0][:-(net_cfg.LSTM.NUM_STACK)];

	for iid in iid_list:
		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			objfeature = np.load(os.path.join(objFeatureDir, "%d.npy" % iid));
			processed_ims.append(objfeature);
		# obj_train.append(np.reshape(processed_ims, (4096)));
		obj_train.append(np.reshape(processed_ims, (timesteps,4096)));

		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			mtnfeature = np.load(os.path.join(mtnFeatureDir, "%d.npy" % iid));
			processed_ims.append(mtnfeature);
		# mtn_train.append(np.reshape(processed_ims, (4096)));
		mtn_train.append(np.reshape(processed_ims, (timesteps,4096)));

	act_label = db.xid2acid[xid];
	mtn_label = db.xid2mcid[xid];
	obj_label = db.xid2ocid[xid];

	print obj_label, mtn_label, act_label, iid_list

def modelPredictLabel(db):
	# history = History()
	hist_list = [];


	# dataSetup_length(db);

	for testid in [3]:
		data_dim = 4096
		timesteps = 10

		obj_train, mtn_train, obj_test, mtn_test, y_train_act, y_test_act, y_train_obj, y_test_obj, y_train_mtn, y_test_mtn, test_labels_act, test_labels_obj, test_labels_mtn = dataSetup(db, timesteps, testid);
		print "Loaded Data"


		modelPath = '/home/bbu/Workspace/data/result/GTEA/models/lstm/multi_task/dropout/active_%d_1_10_fc7_dropout_model_config.json' % (testid);
		weightPath = '/home/bbu/Workspace/data/result/GTEA/models/lstm/multi_task/dropout/active_%d_1_10_fc7_dropout_model_weight.h5' % (testid);

		json_data=open(modelPath).read()

		model = model_from_json(json_data)
		model.load_weights(weightPath)

		classes = model.predict({'object_input':obj_test, 'motion_input': mtn_test}, batch_size=32)

		obj_predict = np.argmax(classes[0], axis=1)
		mtn_predict = np.argmax(classes[1], axis=1)
		act_predict = np.argmax(classes[2], axis=1)

		print np.array(classes[0]).shape
		print np.array(classes[1]).shape
		# print np.array(classes[:,0]).shape
		np.save('data/result/confusion_matrix/predict_result_%d.npy' % testid, np.array([test_labels_obj, test_labels_mtn, test_labels_act, obj_predict, mtn_predict, act_predict]))

		np.savetxt('data/result/confusion_matrix/predict_result_%d.txt' % testid, np.array([test_labels_obj, test_labels_mtn, test_labels_act, obj_predict, mtn_predict, act_predict]), delimiter=',')


		print y_train_act.shape, y_test_act.shape, y_train_obj.shape, y_test_obj.shape, y_train_mtn.shape, y_test_mtn.shape


def modelTestLengthSequence(db):
	hist_list = [];

	timestep = 5;

	for testid in [3]:
		long_xids, short_xids, cnt = dataSetup_length(db, timestep, testid);
		xid2InputData(db, testid, timestep, 68);
		modelPath = '/home/bbu/Workspace/data/result/GTEA/models/fully/multi_task/dropout/active_%d_10_1_dropout_model_config.json' % testid;
		weightPath = '/home/bbu/Workspace/data/result/GTEA/models/fully/multi_task/dropout/active_%d_10_1_dropout_model_weight.h5' % testid;
		json_data=open(modelPath).read()

		model = model_from_json(json_data)
		model.load_weights(weightPath)

		classes = model.predict({'object_input':obj_test, 'motion_input': mtn_test}, batch_size=32)

		# print long_xids, short_xids


def xidsToInput(test_xids):


	test_iid_list = [np.where(db.iid2xid == xid)[0][:-(net_cfg.LSTM.NUM_STACK)] for xid in test_xids];

	test_iids = np.array([]);

	for iid_list in test_iid_list:
		test_iids = np.concatenate((test_iids, iid_list));

	iids = test_iids.astype(int)

	testid = 3;

	objFeatureDir = "/home/bbu/Workspace/data/result/%s/Objectfeature/S_3/fc7" % db.name;
	mtnFeatureDir = "/home/bbu/Workspace/data/result/%s/Motionfeature/S_%d_1/fc7" % (db.name, testid);

	timesteps = 10;

	obj_test = [];
	mtn_test = [];

	for iid in iids:
		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			objfeature = np.load(os.path.join(objFeatureDir, "%d.npy" % iid));
			processed_ims.append(objfeature);
		# obj_test.append(np.reshape(processed_ims, (2048)));
		obj_test.append(np.reshape(processed_ims, (timesteps,2048)));

		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			mtnfeature = np.load(os.path.join(mtnFeatureDir, "%d.npy" % iid));
			processed_ims.append(mtnfeature);
		# mtn_test.append(np.reshape(processed_ims, (4096)));
		mtn_test.append(np.reshape(processed_ims, (timesteps,4096)));

	obj_test = np.array(obj_test);
	mtn_test = np.array(mtn_test);

	return obj_test, mtn_test, iids

def videoLabel(db):

	# xids = set([]);
	# # iids = [];

	# for i in xrange(len(db.iid2xid)):
	# 	if np.char.find(db.iid2path[i], "S4_Coffee_C1") >= 0:
	# 		xids.add(int(db.iid2xid[i]));

	# xids = [ x for x in iter(xids)]
	# print xids

	testid = 3;

	xids = np.where(db.xid2sid == testid)[0]

	# print xids
	# iids = iids[:-net_cfg.LSTM.NUM_STACK]

	obj_test, mtn_test, iids = xidsToInput(xids);

	print "Input Data Loaded"

	print obj_test.shape, mtn_test.shape

	# modelPath = '/home/bbu/Workspace/data/result/GTEA/models/fully/multi_task/dropout/active_%d_10_1_dropout_model_config.json' % testid;
	# weightPath = '/home/bbu/Workspace/data/result/GTEA/models/fully/multi_task/dropout/active_%d_10_1_dropout_model_weight.h5' % testid;
	modelPath = '/home/bbu/Workspace/data/result/GTEA/models/lstm/multi_task/dropout/active_%d_1_10_fc7_dropout_model_config.json' % testid;
	weightPath = '/home/bbu/Workspace/data/result/GTEA/models/lstm/multi_task/dropout/active_%d_1_10_fc7_dropout_model_weight.h5' % testid;

	json_data=open(modelPath).read()

	model = model_from_json(json_data)
	model.load_weights(weightPath)

	print "Model Loaded"

	classes = model.predict({'object_input':obj_test, 'motion_input': mtn_test}, batch_size=32)

	obj_predict = np.argmax(classes[0], axis=1)
	mtn_predict = np.argmax(classes[1], axis=1)
	act_predict = np.argmax(classes[2], axis=1)

	print act_predict
	print act_predict.shape

	np.savetxt('data/result/video_label_1.txt', np.array([act_predict, np.array(iids)]), delimiter=',')




if __name__ == "__main__":
	setting = edict();
	setting.gpu 			= 1;
	setting.db 				= cfg.db.gtea;

	db = Db(setting.db, cfg.dstDir);
	db.genDb();

	KTF.set_session(get_session())

	# modelTestLengthSequence(db);
	modelPredictLabel(db);
	# videoLabel(db);

