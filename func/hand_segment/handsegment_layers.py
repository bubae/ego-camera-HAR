import caffe
import numpy as np
import os, sys
import yaml
from config import cfg
from utils.blob import im_list_to_blob, prep_im_for_blob
import cv2

np.set_printoptions(threshold='nan')


def get_minibatch(db):
	num_images = len(db)

	processed_ims = []
	processed_labels = []
	for i in xrange(num_images):
		im = cv2.imread(db[i]['image'])
		im = im.astype(np.float32, copy=False)
		im -= cfg.TRAIN.PIXEL_MEANS

		processed_ims.append(im)

		label_im = cv2.imread(db[i]['label'])
		label_im = np.mean(label_im, axis=2);
		label = np.zeros_like(label_im, dtype=np.uint8)
		label[label_im > 0] = 1
		# label = label + 1.0;

		# print label
		# label = label[:,:,0] / 255;

		# label = label;
		# print num_images
		# print db[i]['image']
		# print db[i]['label']
		processed_labels.append(label)

	im_blobs = im_list_to_blob(processed_ims)
	# label_blobs = im_list_to_blob(processed_labels)

	max_shape = np.array([im.shape for im in processed_labels]).max(axis=0)
	label_blobs = np.zeros((num_images, max_shape[0], max_shape[1]),dtype=np.float32)

	for i in xrange(num_images):
		label = processed_labels[i]
		label_blobs[i, 0:label.shape[0], 0:label.shape[1]] = label

	blobs = {'data': im_blobs}
	blobs['labels'] = label_blobs

	return blobs


class DataLayer(caffe.Layer):

	def set_db(self, db):
		print 'Set db'
		self._db = db;
		self._numData = len(db);
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
		self._perm = np.random.permutation(np.arange(len(self._db)))
		self._cur = 0;

	def _get_next_batch_inds(self):
		if self._cur + cfg.TRAIN.IMS_PER_BATCH >= len(self._db):
			self._shuffle_db_inds();

		db_inds = self._perm[self._cur:self._cur + cfg.TRAIN.IMS_PER_BATCH]
		self._cur += cfg.TRAIN.IMS_PER_BATCH
		return db_inds

	def _get_blobs(self):
		db_inds = self._get_next_batch_inds();
		batch_db = [self._db[i] for i in db_inds]

		return get_minibatch(batch_db)

	def setup(self, bottom, top):
		print 'setup'

		self._name_to_top_map = {}
		# layer_params = yaml.load(self.param_str_)

		idx = 0
		top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, 3, max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)

		self._name_to_top_map['data'] = idx
		idx +=1

		top[idx].reshape(cfg.TRAIN.IMS_PER_BATCH, max(cfg.TRAIN.SCALES), cfg.TRAIN.MAX_SIZE)
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