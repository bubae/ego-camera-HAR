import os, sys
import caffe
import numpy as np
# from object_net.config import cfg as net_config
from motion_net.config import cfg as net_config
import yaml
from data_layer.minibatch import get_minibatch
from multiprocessing import Process, Queue

class DataLayer(caffe.Layer):

	def set_db(self, db):
		print 'Set db'
		self._db = db;
		self._numData = len(np.where(self._db.iid2tid == 2)[0]);
		self._labels = [];
		labels = np.array([]);

		vlen = [x.shape[0] for x in db.vid2albl];

		for i in xrange(len(db.vid2name)):
			labels = np.concatenate((labels, db.vid2albl[i]));

		self._labels = labels;

		self._shuffle_db_inds()

	def _shuffle_db_inds(self):
		self._perm = np.random.permutation(np.where(self._db.iid2tid == 2)[0])
		self._cur = 0;

	def _get_next_minibatch_inds(self):
		"""Return the roidb indices for the next minibatch."""
		# print self._cur, cfg.TRAIN.IMS_PER_BATCH, len(self._db), len(self._perm)
		# print self._labels

		if self._cur + net_config.TRAIN.IMS_PER_BATCH >= self._numData:
			self._shuffle_db_inds()

		db_inds = self._perm[self._cur:self._cur + net_config.TRAIN.IMS_PER_BATCH]
		self._cur += net_config.TRAIN.IMS_PER_BATCH
		return db_inds

	def _get_next_minibatch(self):
		db_inds = self._get_next_minibatch_inds()
		# minibatch_db = [self._db[i] for i in db_inds]
		return get_minibatch(self._db, db_inds, self._labels)

	def setup(self, bottom, top):
		print 'setup'
		layer_params = yaml.load(self.param_str_)

		self._num_classes = layer_params['num_classes']

		self._name_to_top_map = {}

		idx = 0
		top[idx].reshape(net_config.TRAIN.IMS_PER_BATCH, 3,
			max(net_config.TRAIN.SCALES), net_config.TRAIN.MAX_SIZE)		

		self._name_to_top_map['data'] = idx
		idx += 1
		top[idx].reshape(net_config.TRAIN.IMS_PER_BATCH);
		self._name_to_top_map['labels'] = idx
		idx += 1

		print 'DataLayer: name_to_top:', self._name_to_top_map, len(top)
		assert len(top) == len(self._name_to_top_map)

	def forward(self, bottom, top):
		"""Get blobs and copy them into this layer's top blob vector."""
		print "forward!!"
		blobs = self._get_next_minibatch()

		for blob_name, blob in blobs.iteritems():
			top_ind = self._name_to_top_map[blob_name]
			# Reshape net's input blobs
			top[top_ind].reshape(*(blob.shape))
			# Copy data into net's input blobs
			top[top_ind].data[...] = blob.astype(np.float32, copy=False)

	def backward(self, top, propagate_down, bottom):
		"""This layer does not propagate gradients."""
		pass

	def reshape(self, bottom, top):
		"""Reshaping happens during the call to forward."""
		print "reshape!!"
		pass