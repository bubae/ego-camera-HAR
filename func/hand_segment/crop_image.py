import init_path
import caffe
import numpy as np
import os, sys
from Db import Db
from init_path import cfg
from easydict import EasyDict as edict
from config import cfg as net_cfg
import cv2
import math

# BETA = 5;
# NETWORK_SIZE = 2;
# PIXEL_MEANS = 0;


def getROIBox(ori_img, mask_img, width):

	scale_x = float(ori_img.shape[1]) / float(mask_img.shape[1])
	scale_y = float(ori_img.shape[0]) / float(mask_img.shape[0])

	total_pixel = mask_img.shape[0] * mask_img.shape[1]

	center_point = np.array(mask_img.shape) / 2;
	center_point = center_point[::-1];
	center_point = center_point.astype(np.float32, copy=False);

	a = mask_img > 0

	numPixel = len(mask_img[a]);

	if numPixel == 0:
		point = center_point
	else:
		point = np.array([0.0, 0.0]);

		min_y = 999999999;

		for (y, x), element in np.ndenumerate(a):
			if element:
				if y < min_y:
					min_y = y;

		min_height = 203 - min_y;

		threshold = min_height * net_cfg.CROP.BETA;

		threshold = int(net_cfg.CROP.NETWORK_SIZE[1] -threshold)

		cnt = 0;
		for (y, x), element in np.ndenumerate(a):
			if element and y < threshold:
				point = point + [y, x];
				cnt = cnt + 1;

		# for (x, y), element in np.ndenumerate(a):
		# 	if element:
		# 		point = point + [x, y];

		point = point.astype(np.float32, copy=False) / float(cnt);

		# point = point.astype(np.uint32)

		point = point[::-1];

	# print point

	lamda = math.exp(-net_cfg.CROP.ALPHA * float(numPixel) / float(total_pixel));

	roi_point = (lamda) * point + (1-lamda) * center_point;

	print lamda, roi_point

	scaled_point = np.array([roi_point[0] * scale_x, roi_point[1] * scale_y])

	scaled_point = scaled_point.astype(np.uint32);

	# cv2.circle(ori_img,tuple(scaled_point), 5, (0,0,255), -1)
	
	x1 = 0;
	y1 = 0;
	x2 = width;
	y2 = width;

	if scaled_point[0] < (width/2):
		x1 = 0;
		x2 = width;
	elif scaled_point[0] > net_cfg.CROP.ORIGIN_SIZE[0] - (width/2):
		x1 = net_cfg.CROP.ORIGIN_SIZE[0] - width;
		x2 = net_cfg.CROP.ORIGIN_SIZE[0];
	else:
		x1 = scaled_point[0] - (width/2)
		x2 = scaled_point[0] + (width/2)

	if scaled_point[1] < (width/2):
		y1 = 0;
		y2 = width;
	elif scaled_point[1] > net_cfg.CROP.ORIGIN_SIZE[1] - (width/2):
		y1 = net_cfg.CROP.ORIGIN_SIZE[1] - width;
		y2 = net_cfg.CROP.ORIGIN_SIZE[1];
	else:
		y1 = scaled_point[1] - (width/2)
		y2 = scaled_point[1] + (width/2)

	# pt_1 = [x1, y1];

	# pt_2 = [x2, y2];

	# cv2.rectangle(ori_img, tuple(pt_1), tuple(pt_2),(0,255,0),3)	

	# print numPixel, total_pixel, point, roi_point, scaled_point, lamda

	# print ori_img.shape

	return ori_img[y1:y2, x1:x2];
	# return;



prev_path = '1.jpg'
next_path = '2.jpg'

prev_img = cv2.imread(prev_path);
next_img = cv2.imread(next_path);


def cropROIImages(db, setting):
	prototxtPath = 'models/%s/handsegmentnet/test.prototxt' % db.name;
	modelPath = 'data/hand_segmentation_net/%s_20000.caffemodel' % db.name;

	net = caffe.Net(prototxtPath, modelPath, caffe.TEST)
	net.name = 'handSegmentation';

	imgSaveDir = os.path.join(db.dstDir, 'CROP_IMAGE');

	print imgSaveDir
	if not os.path.isdir(imgSaveDir):
		os.makedirs(imgSaveDir);

	for width in [120, 320]:

		imgSaveDir = os.path.join(db.dstDir, 'CROP_IMAGE', 'r_%d' % width);

		if not os.path.isdir(imgSaveDir):
			os.makedirs(imgSaveDir);

		for i in xrange(len(db.iid2path)):
		# for i in [500]:
			origin_im = cv2.imread(db.iid2path[i]);

			# cv2.imwrite('data/test/%d_ori.jpg' % i, im);

			im = origin_im.astype(np.float32, copy=False)
			im -= net_cfg.CROP.PIXEL_MEANS

			im = cv2.resize(im, (360, 203));

			im = im.transpose((2,0,1))

			net.blobs['data'].reshape(1, *im.shape)
			net.blobs['data'].data[...] = im
			net.forward()

			out = net.blobs['score'].data[0].argmax(axis=0)

			mark_img = out * 255.0

			# out = cv2.resize(out, net_cfg.CROP.ORIGIN_SIZE);

			# print i, db.acid2name[db.xid2acid[db.iid2xid[i]]]
			result_img = getROIBox(origin_im, mark_img, width);
			fileSavePath = os.path.join(imgSaveDir, '%d.jpg' % i);

			print "CROP IMAGE: %d / %d" % (i, len(db.iid2path));

			cv2.imwrite(fileSavePath, result_img);