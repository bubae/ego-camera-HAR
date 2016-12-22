import caffe
import numpy as np
import os, sys
import yaml
from config import cfg
import cv2

np.set_printoptions(threshold='nan')

class Debug_layer(caffe.Layer):

	def setup(self, bottom, top):
		print 'setup'

	def forward(self, bottom, top):
		data = bottom[0].data
		label = bottom[1].data
		# idx = np.where(data==data.max())[0][0]
		# data.shape

		# print data.shape
		# print data
		# for i in xrange(data.shape[0]):
		# 	idx = np.where(data[i]==data[i].max())[0][0]
		# 	print data[i]
		# 	print idx, label[i]

		# print data.shape, label.shape
		# print label
		# score = bottom[2].data
		# upscore = bottom[3].data

		# print data[0,...].transpose(1,2,0).shape

		# for i in xrange(data.shape[0]):
		# 	print score[i,...].transpose(1,2,0).shape
		# 	cv2.imwrite('temp/img_%d_0.jpg' % i, data[i,...].transpose(1,2,0)[:,:,0]);
		# 	cv2.imwrite('temp/img_%d_1.jpg' % i, data[i,...].transpose(1,2,0)[:,:,1]);

		# print data.shape
		# print label.shape
		# print score.shape
		# print upscore.shape

		# print score[0].argmax(axis=0)
		# print score[0,0,:,:].sum()
		# print score[0,1,:,:].sum()
		# print score[0,0,:,:]
		# print score[0,1,:,:]
		# blobs = self._get_blobs();

		# for blob_name, blob in blobs.iteritems():
		# 	top_ind = self._name_to_top_map[blob_name]
		# 	top[top_ind].reshape(*(blob.shape))
		# 	top[top_ind].data[...] = blob.astype(np.float32, copy=False)

	def backward(self, top, propagate_down, bottom):
		"""This layer does not propagate gradients."""
		pass

	def reshape(self, bottom, top):
		"""Reshaping happens during the call to forward."""
		pass