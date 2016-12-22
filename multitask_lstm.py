# from keras.layers import Input, Embedding, LSTM, Dense, merge
# from keras.models import Model

# # headline input: meant to receive sequences of 100 integers, between 1 and 10000.
# # note that we can name any layer by passing it a "name" argument.
# main_input = Input(shape=(100,), dtype='int32', name='main_input')

# # this embedding layer will encode the input sequence
# # into a sequence of dense 512-dimensional vectors.
# x = Embedding(output_dim=512, input_dim=10000, input_length=100)(main_input)

# # a LSTM will transform the vector sequence into a single vector,
# # containing information about the entire sequence
# lstm_out = LSTM(32)(x)

# auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)

# auxiliary_input = Input(shape=(5,), name='aux_input')
# x = merge([lstm_out, auxiliary_input], mode='concat')

# # we stack a deep fully-connected network on top
# x = Dense(64, activation='relu')(x)
# x = Dense(64, activation='relu')(x)
# x = Dense(64, activation='relu')(x)

# # and finally we add the main logistic regression layer
# main_output = Dense(1, activation='sigmoid', name='main_output')(x)

# model = Model(input=[main_input, auxiliary_input], output=[main_output, auxiliary_output])

# model.compile(optimizer='rmsprop', loss='binary_crossentropy',
#               loss_weights=[1., 0.2])

# # model.fit([headline_data, additional_data], [labels, labels],
# #           nb_epoch=50, batch_size=32)


import init_path
from easydict import EasyDict as edict
from init_path import cfg
from Db import Db
import numpy as np 
import os, sys
import tensorflow as tf
tf.python.control_flow_ops = tf
import keras.backend.tensorflow_backend as KTF
from keras.models import Sequential, Model
from keras.layers import Input, merge, LSTM, Dense, Activation, Dropout
from keras.callbacks import History 
from keras.optimizers import SGD
from sklearn.decomposition import PCA
from sklearn import svm
# from sklearn.model_selection import train_test_split
from sendMail import send_mail
from motion_net.config import cfg as net_cfg
import glob

def get_session(gpu_fraction=0.95):
	'''Assume that you have 6GB of GPU memory and want to allocate ~2GB'''

	num_threads = os.environ.get('OMP_NUM_THREADS')
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)

	if num_threads:
		return tf.Session(config=tf.ConfigProto(
			gpu_options=gpu_options, intra_op_parallelism_threads=num_threads))
	else:
		return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


## LSTM Merged
def build_model_1(timesteps, data_dim, nb_classes):

	object_input = Input(shape=(timesteps, 4096), name='object_input')

	# obj_x = LSTM(2048)(object_input)

	# obj_x = Dropout(0.5)(obj_x);

	# object_output = Dense(db.xid2ocid.max()+1, activation='softmax', name='object_output')(obj_x);

	motion_input = Input(shape=(timesteps, data_dim), name='motion_input')

	# mtn_x = LSTM(4096)(motion_input);

	# mtn_x = Dropout(0.5)(mtn_x);

	# motion_output = Dense(db.xid2mcid.max()+1, activation='softmax', name='motion_output')(mtn_x);


	action_input = merge([object_input, motion_input], mode='concat');

	act_lstm_1 = LSTM(4096, return_sequences=True, stateful=True)(action_input)

	act_lstm_2 = LSTM(4096, stateful=True)(act_lstm_1)


	action_output = Dense(nb_classes, activation='softmax', name='action_output')(act_lstm_2);


	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


	model = Model(input=[object_input, motion_input], output=action_output)

	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              loss_weights=[0.2,0.2,0.8],
	              metrics=['accuracy'])

	return model


## Fully Connected
def build_model_2(timesteps, data_dim, nb_classes):

	object_input = Input(shape=(timesteps, data_dim), name='object_input')

	obj_x = Dense(4096)(object_input)

	obj_x = Dense(4096, activation='relu')(object_input)


	object_output = Dense(db.xid2ocid.max()+1, activation='softmax', name='object_output')(obj_x);

	motion_input = Input(shape=(timesteps, data_dim), name='motion_input')

	mtn_x = LSTM(4096)(motion_input);

	motion_output = Dense(db.xid2mcid.max()+1, activation='softmax', name='motion_output')(mtn_x);


	action_input = merge([obj_x, mtn_x], mode='concat');

	action_output = Dense(nb_classes, activation='softmax', name='action_output')(action_input);


	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


	model = Model(input=[object_input, motion_input], output=[object_output, motion_output, action_output])

	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              loss_weights=[0.2,0.1,0.8],
	              metrics=['accuracy'])

	object_model = Sequential()
	object_model.add(Dense(4096, input_dim=data_dim))
	# object_model.add(Activation('relu'))
	# object_model.add(Dropout(0.5))
	# object_model.add(LSTM(4096, input_shape=(timesteps, data_dim)))

	motion_model = Sequential()
	motion_model.add(Dense(4096, input_dim=data_dim))
	# motion_model.add(Activation('relu'))
	# motion_model.add(Dropout(0.5))
	# motion_model.add(LSTM(4096, input_shape=(timesteps, data_dim)))


	merge_model = Sequential()
	merge_model.add(Merge([object_model, motion_model], mode='concat'))
	# merge_model.add(LSTM(4096, return_sequences=True, input_shape=(timesteps, data_dim)))
	# merge_model.add(Dropout(setting.dropout))
	# merge_model.add(LSTM(4096, input_shape=(timesteps, data_dim)))

	# merge_model.add(Dropout(setting.dropout))
	merge_model.add(Dense(nb_classes, activation='softmax'))
	# model.add(LSTM(32, return_sequences=True,
	#                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
	# model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
	# model.add(LSTM(32))  # return a single vector of dimension 32
	# model.add(Dense(nb_classes, activation='softmax'))

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

	merge_model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              metrics=['accuracy'])

	print "LSTM Complie Finished"

	return merge_model


def build_model_3(timesteps, data_dim, nb_classes):

	object_model = Sequential()
	object_model.add(Dense(4096, input_dim=data_dim))
	# object_model.add(Activation('relu'))
	# object_model.add(Dropout(0.5))
	# object_model.add(LSTM(4096, input_shape=(timesteps, data_dim)))

	motion_model = Sequential()
	motion_model.add(Dense(4096, input_dim=data_dim))
	# motion_model.add(Activation('relu'))
	# motion_model.add(Dropout(0.5))
	# motion_model.add(LSTM(4096, input_shape=(timesteps, data_dim)))


	merge_model = Sequential()
	merge_model.add(Merge([object_model, motion_model], mode='concat'))
	# merge_model.add(LSTM(4096, return_sequences=True, input_shape=(timesteps, data_dim)))
	# merge_model.add(Dropout(setting.dropout))
	# merge_model.add(LSTM(4096, input_shape=(timesteps, data_dim)))

	# merge_model.add(Dropout(setting.dropout))
	merge_model.add(Dense(nb_classes, activation='softmax'))
	# model.add(LSTM(32, return_sequences=True,
	#                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
	# model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
	# model.add(LSTM(32))  # return a single vector of dimension 32
	# model.add(Dense(nb_classes, activation='softmax'))

	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

	merge_model.compile(loss='categorical_crossentropy',
	              optimizer='adam',
	              metrics=['accuracy'])

	print "LSTM Complie Finished"

	return merge_model



def build_model_4(timesteps, data_dim, nb_classes):

	object_input = Input(shape=(data_dim,), name='object_input')

	obj_x = Dense(4096, activation='relu')(object_input)

	obj_x = Dropout(0.5)(obj_x);

	object_output = Dense(db.xid2ocid.max()+1, activation='softmax', name='object_output')(obj_x);

	motion_input = Input(shape=(data_dim,), name='motion_input')

	mtn_x = Dense(4096, activation='relu')(motion_input);
	mtn_x = Dropout(0.5)(mtn_x);
	motion_output = Dense(db.xid2mcid.max()+1, activation='softmax', name='motion_output')(mtn_x);


	action_input = merge([obj_x, mtn_x], mode='concat');

	action_output = Dense(nb_classes, activation='softmax', name='action_output')(action_input);


	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


	model = Model(input=[object_input, motion_input], output=[object_output, motion_output, action_output])

	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              loss_weights=[0.2,0.2,0.8],
	              metrics=['accuracy'])

	return model

def build_model(timesteps, data_dim, nb_classes):

	object_input = Input(shape=(timesteps, 2048), name='object_input')

	obj_x = LSTM(2048, return_sequences=True)(object_input)

	obj_x_1 = Dropout(0.5)(obj_x);

	obj_x_2 = LSTM(2048)(obj_x_1);
	
	obj_x_3 = Dropout(0.5)(obj_x_2);

	object_output = Dense(db.xid2ocid.max()+1, activation='softmax', name='object_output')(obj_x_3);

	motion_input = Input(shape=(timesteps, data_dim), name='motion_input')

	mtn_x = LSTM(4096, return_sequences=True)(motion_input);

	mtn_x_1 = Dropout(0.5)(mtn_x);

	mtn_x_2 = LSTM(4096)(mtn_x_1);

	mtn_x_3 = Dropout(0.5)(mtn_x_2);

	motion_output = Dense(db.xid2mcid.max()+1, activation='softmax', name='motion_output')(mtn_x_3);

	action_input = merge([obj_x_3, mtn_x_3], mode='concat');

	action_output = Dense(nb_classes, activation='softmax', name='action_output')(action_input);


	sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)


	model = Model(input=[object_input, motion_input], output=[object_output, motion_output, action_output])

	model.compile(loss='categorical_crossentropy',
	              optimizer=sgd,
	              loss_weights=[0.2,0.2,0.8],
	              metrics=['accuracy'])

	return model


def dataSetup(db, timesteps, testid):

	objFeatureDir = "/home/bbu/Workspace/data/result/%s/Objectfeature/S_%d/fc7" % (db.name, testid);
	mtnFeatureDir = "/home/bbu/Workspace/data/result/%s/Motionfeature/S_%d_1/fc7" % (db.name, testid);

	imgList = glob.glob("/home/bbu/Workspace/data/result/%s/Motionfeature/S_%d_1/fc7/*_flip.npy" % (db.name, testid));

	iids = np.array([int(os.path.basename(path)[:-9]) for path in imgList]);
	iids = np.sort(iids);
	labels = np.array([int(db.xid2acid[db.iid2xid[iid]]) for iid in iids])

	train_xids = np.where(db.xid2sid != testid)[0];

	# train_iids = np.array([iid for iid in iids if len(np.where(train_xids == db.iid2xid[iid])[0]) > 0])

	train_iid_list = [np.where(db.iid2xid == xid)[0][:-(net_cfg.LSTM.NUM_STACK)] for xid in train_xids];

	train_iids = np.array([]);
	train_iids = train_iids.astype(int)

	for iid_list in train_iid_list:
		train_iids = np.concatenate((train_iids, iid_list));

	test_xids = np.where(db.xid2sid == testid)[0];

	# test_iids = np.array([iid for iid in iids if len(np.where(test_xids == db.iid2xid[iid])[0]) > 0])

	# print train_iids.shape, test_iids.shape

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
			objfeature = np.load(os.path.join(objFeatureDir, "%d.npy" % i));
			processed_ims.append(objfeature);
		# obj_train.append(np.reshape(processed_ims, (4096)));
		obj_train.append(np.reshape(processed_ims, (timesteps,2048)));

		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			mtnfeature = np.load(os.path.join(mtnFeatureDir, "%d.npy" % i));
			processed_ims.append(mtnfeature);
		# mtn_train.append(np.reshape(processed_ims, (4096)));
		mtn_train.append(np.reshape(processed_ims, (timesteps,4096)));

	for iid in test_iids:
		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			objfeature = np.load(os.path.join(objFeatureDir, "%d.npy" % i));
			processed_ims.append(objfeature);
		# obj_test.append(np.reshape(processed_ims, (4096)));
		obj_test.append(np.reshape(processed_ims, (timesteps, 2048)));

		processed_ims = [];
		for i in xrange(iid, iid+timesteps):
			mtnfeature = np.load(os.path.join(mtnFeatureDir, "%d.npy" % i));
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
	test_labels = db.xid2acid[db.iid2xid[test_iids].astype(int)].astype(int)

	y_train_act = np.zeros((len(train_iids), max(db.xid2acid)+1))
	y_train_act[np.arange(len(train_iids)), train_labels] = 1
	y_test_act = np.zeros((len(test_iids), max(db.xid2acid)+1))
	y_test_act[np.arange(len(test_iids)), test_labels] = 1

	train_labels = db.xid2ocid[db.iid2xid[train_iids].astype(int)].astype(int)
	test_labels = db.xid2ocid[db.iid2xid[test_iids].astype(int)].astype(int)

	y_train_obj = np.zeros((len(train_iids), max(db.xid2ocid)+1))
	y_train_obj[np.arange(len(train_iids)), train_labels] = 1
	y_test_obj = np.zeros((len(test_iids), max(db.xid2ocid)+1))
	y_test_obj[np.arange(len(test_iids)), test_labels] = 1

	train_labels = db.xid2mcid[db.iid2xid[train_iids].astype(int)].astype(int)
	test_labels = db.xid2mcid[db.iid2xid[test_iids].astype(int)].astype(int)

	y_train_mtn = np.zeros((len(train_iids), max(db.xid2mcid)+1))
	y_train_mtn[np.arange(len(train_iids)), train_labels] = 1
	y_test_mtn = np.zeros((len(test_iids), max(db.xid2mcid)+1))
	y_test_mtn[np.arange(len(test_iids)), test_labels] = 1	


	return obj_train, mtn_train, obj_test, mtn_test, y_train_act, y_test_act, y_train_obj, y_test_obj, y_train_mtn, y_test_mtn
	# for i in xrange(len(iid2xid)):
	# 	feature = np.load(os.path.join(objFeaturePath, "%d.npy" % i));
	# 	X.append(np.reshape(feature, 4096));
	# 	# Y_test.append(label[test_iids[i]])

	# X = np.array(X);
	# X_train = np.array([X[iids:iids+net_cfg.TRAIN.NUM_STACK,:] for iids in train_iids])
	# Y_train_iids = np.array([int(label[iids]) for iids in train_iids])

	# Y_train = np.zeros((len(Y_train_iids), max(db.xid2acid)+1))
	# Y_train[np.arange(len(Y_train_iids)), Y_train_iids] = 1

	# # print Y_train
	# X_test = np.array([X[iids:iids+net_cfg.TRAIN.NUM_STACK,:] for iids in test_iids]);
	# Y_test_iids = np.array([int(label[iids]) for iids in test_iids]);

	# Y_test = np.zeros((len(Y_test_iids), max(db.xid2acid)+1))
	# Y_test[np.arange(len(Y_test_iids)), Y_test_iids] = 1

	# iids = iids.sort();
	# featureDIr = 

	# print labels

def modelTrain(db):
	# history = History()
	hist_list = [];

	dstDir = os.path.join(db.dstDir, 'models', 'lstm', 'multi_task', 'dropout_adv');

	if not os.path.exists(dstDir):
		os.makedirs(dstDir);

	# for timesteps in [5]:
	for testid in [2,3]:
		data_dim = 4096
		timesteps = 10

		obj_train, mtn_train, obj_test, mtn_test, y_train_act, y_test_act, y_train_obj, y_test_obj, y_train_mtn, y_test_mtn = dataSetup(db, timesteps, testid);
		print "Loaded Data"

		print y_train_act.shape, y_test_act.shape, y_train_obj.shape, y_test_obj.shape, y_train_mtn.shape, y_test_mtn.shape
		# print obj_train.shape, mtn_train.shape, obj_test.shape, mtn_test.shape, y_train.shape, y_test.shape, len(obj_train)
		# # dataSetup(db, 0);
		num_train = len(obj_train)
		num_test = len(obj_test)

		nb_classes = max(db.xid2acid)+1;

		print "Building Model"
		# # model = build_model(data_dim, nb_classes);

		model = build_model(timesteps, data_dim, nb_classes);

		# # print num_train, num_test, data_dim, timesteps, nb_classes

		# # x_val_a = np.random.random((num_test, timesteps, data_dim))
		# # x_val_b = np.random.random((num_test, timesteps, data_dim))
		# # y_val = np.random.random((num_test, nb_classes))

		print "Train LSTM"
		# # model.fit(obj_train, y_train,
		# # 			batch_size=32, nb_epoch=20,
		# # 			validation_data=(obj_test, y_test))

		history = model.fit({'object_input':obj_train, 'motion_input': mtn_train}, {'object_output':y_train_obj, 'motion_output':y_train_mtn, 'action_output': y_train_act},
					batch_size=32, nb_epoch=5,
					validation_data=({'object_input':obj_test, 'motion_input': mtn_test}, {'object_output':y_test_obj, 'motion_output':y_test_mtn, 'action_output': y_test_act}))

		fileNameH5 = "active_%d_1_%d_dropout_adv_model_weight.h5" % (testid, timesteps)
		fileNameJSON = "active_%d_1_%d_dropout_adv_model_config.json" % (testid, timesteps)

		filePath = os.path.join(dstDir, fileNameH5);
		model.save_weights(filePath, overwrite=True)
		filePath = os.path.join(dstDir, fileNameJSON);
		open(filePath, 'w').write(model.to_json());

		# # hist_list.append(history.history);
		print history.history
		np.savetxt('data/result/lstm_result/multi_task/lstm/lstm_%d_1_%d_dropout_adv.txt' % (testid, timesteps), np.array([history.history['val_action_output_acc'], history.history['val_motion_output_acc'], history.history['val_object_output_acc']]), delimiter=',')


	# print hist_list
	# hist_list = np.hist_list(hist_list);
	# np.savetxt('data/result/lstm_oneimage_fivelength.txt', hist_list, delimiter=',')

if __name__ == "__main__":
	setting = edict();
	setting.gpu 			= 1;
	setting.db 				= cfg.db.gtea;

	db = Db(setting.db, cfg.dstDir);
	db.genDb();

	KTF.set_session(get_session())

	modelTrain(db);

	# print X_train.shape
	# print X_test.shape
	# mergeLayerTest(db, X_train, [], X_test, [], Y_train, Y_test)
