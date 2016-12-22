import init_path
import caffe
import cv2, os, sys
from utils.blob import im_list_to_blob, prep_im_for_blob
from config import cfg
import numpy as np 
DATA_DIR = 'data/test'

def get_blob(im):
	print im.shape
	im = im.astype(np.float32, copy=False)
	im -= cfg.TRAIN.PIXEL_MEANS

	im = cv2.resize(im, (cfg.TRAIN.MAX_SIZE, cfg.TRAIN.SCALES[0]))

	# im = im / 255
	# # print im
	# cv2.imshow('frame', im)
	# cv2.waitKey(0);
	# sys.exit(1)
	cv2.imwrite('temp/haha.jpg', im);

	im_list = []
	im_list.append(im)

	im_blobs = im_list_to_blob(im_list);

	print im_blobs.shape

	im_blobs = im_blobs.astype(np.float32, copy=False);

	blob = {'data': im_blobs}

	return blob

def test_net(net, im):

	# forward_kwargs = {'data': blob['data'].astype(np.float32, copy=False)}
	im = cv2.resize(im, (cfg.TRAIN.MAX_SIZE, cfg.TRAIN.SCALES[0]))
	im = im.astype(np.float32, copy=False)
	im -= cfg.TRAIN.PIXEL_MEANS
	im = im.transpose((2,0,1))

	net.blobs['data'].reshape(1, *im.shape)
	net.blobs['data'].data[...] = im
	net.forward()

	out = net.blobs['score'].data[0].argmax(axis=0)

	# print out
	# print out.shape

	return out
	# in_ = in_.transpose((2,0,1))

	# blobs_out = net.forward(**blob);

	# print blobs_out['score'].shape

	# print blobs_out['score']
	# return blobs_out['score']

	# blob = blob.transpose(channel_swap)
	# print blobs_out['score'].shape
	# return blobs_out['score']
