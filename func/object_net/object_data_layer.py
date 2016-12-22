import caffe
import numpy as np
import os, sys
import yaml
from config import cfg
from utils.blob import im_list_to_blob, prep_im_for_blob
import cv2

def get_minibatch(train_iids, isFlip, db, width):
	num_images = len(train_iids);

	processed_ims = []
	processed_labels = []

	# frameDir = '/home/bbu/Workspace/data/result/%s/CROP_IMAGE/r_%d' % (db.name, width);

	frameDir = '/home/bbu/Workspace/data/result/%s/CROP_IMAGE' % db.name;

	for i in xrange(num_images):
		iid = train_iids[i];

		im = cv2.imread(os.path.join(frameDir, '%d.jpg' % iid));
		# im = cv2.imread(db.iid2path[iid]);
		# h, w, c = im.shape
		# print h,w,c;
		# h = h / 2;
		# w = w / 2;
		# im = im[h-150:h+150,w-150:w+150, :];
		# print im.shape
		# print im.shape
		im = cv2.resize(im, (cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE), interpolation = cv2.INTER_LINEAR);
		im = im.astype(np.float32, copy=False)
		im -= cfg.TRAIN.PIXEL_MEANS

		if isFlip[i]:
			im = im[:, ::-1, :]

		processed_ims.append(im);

		xid = db.iid2xid[int(iid)];
		ocid = db.xid2ocid[int(xid)];

		processed_labels.append(int(ocid));

	im_blobs = im_list_to_blob(processed_ims);

	label_blobs = np.zeros((num_images, 1), dtype=np.float32);
	for i in xrange(num_images):
		label_blobs[i, 0] = processed_labels[i];

	blobs = {'data': im_blobs}
	blobs['labels'] = label_blobs

	return blobs

class DataLayer(caffe.Layer):

	def set_db(self, db, test_id, width):
		print 'Set db'

		train_task_ids = np.where(db.xid2sid != test_id)[0];

		train_iid_list = [np.where(db.iid2xid == xid)[0] for xid in train_task_ids];

		train_iids = np.array([]);

		for iid_list in train_iid_list:
			train_iids = np.concatenate((train_iids, iid_list));

		# print cfg.TRAIN.PIXEL_MEANS;

		# print train_iids.shape, test_id
		flip_train_iids = np.concatenate((train_iids, train_iids));
		# print flip_train_iids.shape

		self._isFlip = np.zeros(len(flip_train_iids));
		self._isFlip[len(train_iids):] = 1;
		
		# print len(np.where(self._isFlip == 1)[0])

		self._db = db;
		self._train_iids = flip_train_iids;
		self._numData = len(flip_train_iids);
		self._width = width;
		# self._db = db;
		# self._numData = len(np.where(self._db.iid2tid == 2)[0]);
		# self._labels = [];
		# labels = np.array([]);

		# vlen = [x.shape[0] for x in db.vid2albl];

		# for i in xrange(len(db.vid2name)):
		# 	labels = np.concatenate((labels, db.vid2albl[i]));

		# self._labels = labels;

		self._shuffle_db_inds()

	def _shuffle_db_inds(self):
		self._perm = np.random.permutation(np.arange(len(self._train_iids)))
		self._cur = 0;

	def _get_next_batch_inds(self):
		if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._train_iids):
			self._shuffle_db_inds();

		db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
		self._cur += cfg.TRAIN.IMS_PER_BATCH
		return db_inds

	def _get_blobs(self):
		db_inds = self._get_next_batch_inds();
		batch_db = np.array([self._train_iids[i] for i in db_inds]);
		isFlip = np.array([self._isFlip[i] for i in db_inds]);

		return get_minibatch(batch_db, isFlip, self._db, self._width)

	def setup(self, bottom, top):
		print 'setup'

		self._name_to_top_map = {}
		# layer_params = yaml.load(self.param_str_)

		idx = 0
		top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)

		self._name_to_top_map['data'] = idx
		idx +=1

		top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 1)
		self._name_to_top_map['labels'] = idx

	def forward(self, bottom, top):
		blobs = self._get_blobs();

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