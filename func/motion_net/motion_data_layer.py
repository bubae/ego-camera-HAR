import caffe
import numpy as np
import os, sys
import yaml
from config import cfg
from utils.blob import im_list_to_blob, prep_im_for_blob
import cv2


def get_stackImg(iid, isFlip, db):


	ims = [];

	frameDir = '/home/bbu/Workspace/data/result/%s/denseflow_backmotion' % db.name;

	for i in xrange(int(iid), int(iid)+cfg.TRAIN.NUM_STACK):

		# print db.iid2path[i]
		im_x = cv2.imread(os.path.join(frameDir, 'flow_x_%05d.jpg' % i));
		im_x = im_x.astype(np.float32, copy=False)

		grey_im_x = cv2.cvtColor(im_x,cv2.COLOR_BGR2GRAY);
		# print grey_im_x.shape
		grey_im_x = cv2.resize(grey_im_x, (cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE));

		# print grey_im_x.shape
		if isFlip:
			grey_im_x = 255 - grey_im_x[:, ::-1]

		grey_im_x -= cfg.TRAIN.PIXEL_MEANS[0]
		ims.append(grey_im_x);

		im_y = cv2.imread(os.path.join(frameDir, 'flow_y_%05d.jpg' % i));
		im_y = im_y.astype(np.float32, copy=False)

		grey_im_y = cv2.cvtColor(im_y,cv2.COLOR_BGR2GRAY);

		grey_im_y = cv2.resize(grey_im_y, (cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE));

		grey_im_y -= cfg.TRAIN.PIXEL_MEANS[0]

		# cv2.imwrite('temp/%d_x.jpg' % i, grey_im_x);
		# cv2.imwrite('temp/%d_y.jpg' % i, grey_im_y);
		# print grey_im_x.shape

		if isFlip:
			grey_im_y = grey_im_y[:, ::-1]

		ims.append(grey_im_y);

	return np.array(ims);


def get_minibatch(train_iids, isFlip, db):
	num_images = len(train_iids);

	processed_ims = []
	processed_labels = []

	dims = (num_images, 2*cfg.TRAIN.NUM_STACK, 224, 224);

	for i in xrange(num_images):
		iid = train_iids[i];

		# print iid
		stackImg = get_stackImg(iid, isFlip[i], db);

		# print num_images, stackImg.shape
		# im = cv2.imread(os.path.join(frameDir, '%d.jpg' % iid));
		# im = im.astype(np.float32, copy=False)
		# im -= cfg.TRAIN.PIXEL_MEANS

		# if isFlip[i]:
			# im = im[:, ::-1, :]
		# twoStack = np.concatenate((stackImg, stackImg));
		# print twoStack.shape
		# processed_ims.append(twoStack)
		processed_ims.append(stackImg)

		xid = db.iid2xid[int(iid)];
		mcid = db.xid2mcid[int(xid)];

		processed_labels.append(int(mcid))

	im_blobs = np.zeros(shape=dims, dtype=np.float32)

	# im_blobs = im_list_to_blob(processed_ims);

	for i in xrange(num_images):
		im_blobs[i,::] = processed_ims[i];

	# print im_blobs.shape
	# print im_blobs.shape

	label_blobs = np.zeros((num_images, 1), dtype=np.float32);
	for i in xrange(num_images):
		label_blobs[i, 0] = processed_labels[i];

	# print label_blobs.shape

	blobs = {'data': im_blobs}
	blobs['labels'] = label_blobs

	return blobs

class DataLayer(caffe.Layer):

	def set_db(self, db, test_id):
		print 'Set db'

		train_task_ids = np.where(db.xid2sid != test_id)[0];

		train_iid_list = [np.where(db.iid2xid == xid)[0][:-(cfg.TRAIN.NUM_STACK)] for xid in train_task_ids];

		# for xid in train_task_ids:
			# print np.where(db.iid2xid == xid)[0]
			# print np.where(db.iid2xid == xid)[0][:-4]

		train_iids = np.array([]);

		for iid_list in train_iid_list:
			train_iids = np.concatenate((train_iids, iid_list));

		# print train_iids.shape, test_id
		flip_train_iids = np.concatenate((train_iids, train_iids));
		# print flip_train_iids.shape

		self._isFlip = np.zeros(len(flip_train_iids));
		self._isFlip[len(train_iids):] = 1;
		
		# print len(np.where(self._isFlip == 1)[0])

		self._db = db;
		self._train_iids = flip_train_iids;
		self._numData = len(flip_train_iids);
		# self._db = db;
		# self._numData = len(np.where(self._db.iid2tid == 2)[0]);
		# self._labels = [];
		# labels = np.array([]);

		# vlen = [x.shape[0] for x in db.vid2albl];

		# for i in xrange(len(db.vid2name)):
		# 	labels = np.concatenate((labels, db.vid2albl[i]));

		# self._labels = labels;

		self._shuffle_db_inds()
		# self._cur = 0;
		# self._perm = np.arange(len(self._train_iids))

	def _shuffle_db_inds(self):
		self._perm = np.random.permutation(np.arange(len(self._train_iids)))
		self._cur = 0;

	def _get_next_batch_inds(self):
		if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._train_iids):
			self._shuffle_db_inds();
			# self._cur = 0;

		db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
		self._cur += cfg.TRAIN.IMS_PER_BATCH
		return db_inds

	def _get_blobs(self):
		db_inds = self._get_next_batch_inds();
		batch_db = np.array([self._train_iids[i] for i in db_inds]);
		isFlip = np.array([self._isFlip[i] for i in db_inds]);

		return get_minibatch(batch_db, isFlip, self._db), db_inds, batch_db, isFlip

	def setup(self, bottom, top):

		self._name_to_top_map = {}
		# layer_params = yaml.load(self.param_str_)

		idx = 0
		top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, cfg.TRAIN.NUM_STACK*2, max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)

		self._name_to_top_map['data'] = idx
		idx +=1

		top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 1)
		self._name_to_top_map['labels'] = idx

	def forward(self, bottom, top):
		blobs, db_inds, batch_db, isFlip  = self._get_blobs();
		# print blobs['labels'], blobs['data'].shape

		for blob_name, blob in blobs.iteritems():
			top_ind = self._name_to_top_map[blob_name]
			top[top_ind].reshape(*(blob.shape))
			top[top_ind].data[...] = blob.astype(np.float32, copy=False)

	def backward(self, top, propagate_down, bottom):
		"""This layer does not propagate gradients."""
		pass

	def reshape(self, bottom, top):
		"""Reshaping happens during the call to forward."""
		pass