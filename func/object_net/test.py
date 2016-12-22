import numpy as np
import os, sys
import caffe
from object_net.config import cfg as net_cfg
from utils.blob import im_list_to_blob, prep_im_for_blob
import cv2

def get_blob(db, iids):
	num_images = len(iids);

	processed_ims = [];
	processed_labels = [];

	# frameDir = '/home/bbu/Workspace/data/result/%s/CROP_IMAGE/r_%d' % (db.name, width);

	frameDir = '/home/bbu/Workspace/data/result/%s/CROP_IMAGE' % db.name;


	for i in xrange(num_images):
		iid = iids[i];

		im = cv2.imread(os.path.join(frameDir, '%d.jpg' % iid))
		# im = cv2.imread(db.iid2path[iid]);
		# h, w, c = im.shape
		# h = h / 2;
		# w = w / 2;
		# im = im[h-150:h+150,w-150:w+150, :];
		# print im.shape
		# print im.shape
		im = cv2.resize(im, (net_cfg.TRAIN.MAX_SIZE, net_cfg.TRAIN.MAX_SIZE), interpolation = cv2.INTER_LINEAR);
		
		im = im.astype(np.float32, copy=False)
		im -= net_cfg.TRAIN.PIXEL_MEANS

		processed_ims.append(im)

		xid = db.iid2xid[int(iid)];
		ocid = db.xid2ocid[int(xid)];

		processed_labels.append(int(ocid))

	im_blobs = im_list_to_blob(processed_ims);

	label_blobs = np.zeros((num_images, 1), dtype=np.float32);
	for i in xrange(num_images):
		label_blobs[i, 0] = processed_labels[i];

	blobs = {'data': im_blobs}
	blobs['labels'] = label_blobs

	return blobs


def test_net(prototxtPath, modelPath, test_id, db, width=226):

	# accuracy_list = [];
	dstDir = 'data/result/objectNet/S_%d' % test_id;

	if not os.path.exists(dstDir):
		os.makedirs(dstDir);

	accuracy = 0.0;

	net = caffe.Net(prototxtPath, modelPath, caffe.TEST)
	net.name = 'test_object_net'

	test_xids = np.where(db.xid2sid == test_id)[0];

	test_iids_list = [np.where(db.iid2xid == xid)[0] for xid in test_xids];

	test_iids = np.array([]);

	for iid_list in test_iids_list:
		test_iids = np.concatenate((test_iids, iid_list))

	batch_size = net_cfg.TEST.BATCH_SIZE;
	test_iters = int(len(test_iids) / batch_size)

	_cur = 0;

	for _iter in xrange(test_iters):
		blob = get_blob(db, test_iids[_cur:_cur+batch_size]);
		# solver.test_nets[0]

		# net.blobs['data'].reshape(1, *im.shape)
		blobs_out = net.forward(**blob);

		# print blobs_out['accuracy']
		# print blobs_out['accuracy']
		print blobs_out['accuracy']
		accuracy += blobs_out['accuracy'];

		_cur = _cur + batch_size;

		print 'Test %d: %d / %d' % (test_id, _iter, test_iters);

	accuracy /= float(test_iters);

	print accuracy
	# accuracy_list.append(accuracy);

	filePath = os.path.join(dstDir, 'object_net_%d_accuracy.txt' % test_id);

	with open(filePath,'w') as f:
		f.write(str(accuracy))

	return accuracy


def get_y_predict(prototxtPath, modelPath, db, test_id):

	savePath = os.path.join('data/result/confusion_matrix/object/predict_result_%d.npy' % test_id);

	net = caffe.Net(prototxtPath, modelPath, caffe.TEST)
	net.name = 'test_object_net'	

	y_predicts = np.array([]);

	test_xids = np.where(db.xid2sid == test_id)[0];

	test_iids_list = [np.where(db.iid2xid == xid)[0] for xid in test_xids];

	test_iids = np.array([]);

	for iid_list in test_iids_list:
		test_iids = np.concatenate((test_iids, iid_list))

	test_iids = test_iids.astype(int)
	y_target = np.array(db.xid2ocid[db.iid2xid[test_iids].astype(int)].astype(int))

	print y_target.shape;

	cnt = 0;

	for iid in test_iids:
		blob = get_blob(db, [iid]);

		blobs_out = net.forward(**blob);

		score = net.blobs['cls_score'].data.copy();


		y_predict = np.argmax(score, axis=1)

		# print y_predict;

		y_predicts = np.concatenate([y_predicts, y_predict]);

		cnt = cnt + 1;
		print "Predict %d / %d" % (cnt, len(test_iids));

	y_predicts = np.array(y_predicts);

	print y_predicts.shape;

	np.save(savePath, np.array([y_target, y_predicts]));
	


def extract_object_feature(prototxtPath, modelPath, db, testid):
	# prototxtPath = 'models/%s/objectnet/VGG_CNN_M_2048/deploy.prototxt' % db.name
	# modelPath = 'data/snapshot/vgg_cnn_m_2048_iter_4_10000.caffemodel';

	net = caffe.Net(prototxtPath, modelPath, caffe.TEST)
	net.name = 'test_object_net'

	frameDir = '/home/bbu/Workspace/data/result/%s/CROP_IMAGE' % db.name;

	fc6_dstDir = os.path.join(db.dstDir, 'Objectfeature', 'S_%d' % testid, 'fc6');
	fc7_dstDir = os.path.join(db.dstDir, 'Objectfeature', 'S_%d' % testid, 'fc7');

	if not os.path.exists(fc6_dstDir):
		os.makedirs(fc6_dstDir);	

	if not os.path.exists(fc7_dstDir):
		os.makedirs(fc7_dstDir);	

	for iid in xrange(len(db.iid2path)):
		blob = get_blob(db, [iid])

		blobs_out = net.forward(**blob);

		fc6_feature = net.blobs['fc6'].data.copy();
		fc6_featurePath = os.path.join(fc6_dstDir, '%d.npy' % iid);

		fc7_feature = net.blobs['fc7'].data.copy();
		fc7_featurePath = os.path.join(fc7_dstDir, '%d.npy' % iid);

		print 'Feature Extract %d / %d Feature Saved' % (iid, len(db.iid2path));
		np.save(fc6_featurePath, fc6_feature);
		np.save(fc7_featurePath, fc7_feature);