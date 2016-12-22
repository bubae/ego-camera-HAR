import numpy as np
import numpy.random as npr
import cv2
# from object_net.config import cfg as net_config
from motion_net.config import cfg as net_config
# from utils.blob import prep_im_for_blob, im_list_to_blob

def _get_image_blob(db, db_inds):
	num_images = len(db_inds);

	processed_ims = []
	im_scales = []
	for i in db_inds:

		if net_config.TRAIN.FLOWNET:
			im_orig = cv2.imread(db.iid2path[i])
		else:
			im_orig = cv2.imread(db.iid2path[i])

		# if roidb[i]['flipped']:
		# 	im = im[:, ::-1, :]
		im = im_orig.astype(np.float32, copy=False)
		im -= net_config.TRAIN.PIXEL_MEANS
		im_shape = im.shape
		im_scale = [float(net_config.TRAIN.SCALES[0]) / float(im_shape[1]), float(net_config.TRAIN.SCALES[0]) / float(im_shape[0])]

		im = cv2.resize(im, None, None, fx=im_scale[0], fy=im_scale[1], interpolation=cv2.INTER_LINEAR)
		# target_size = net_config.TRAIN.SCALES[0]
		# im, im_scale = prep_im_for_blob(im_orig, net_config.TRAIN.PIXEL_MEANS, target_size, net_config.TRAIN.MAX_SIZE)
		im_scales.append(1)
		processed_ims.append(im)

	# Create a blob to hold the input images
	blob = im_list_to_blob(processed_ims)

	return blob, im_scales

def get_minibatch(db, db_inds, labels):
	num_images =  1

	# blobs['labels'] = labels_blob

	im_blobs, im_scales = _get_image_blob(db, db_inds);

	blobs = {'data': im_blobs}
	blobs['labels'] = labels[db_inds];

	return blobs


def im_list_to_blob(ims):
	"""Convert a list of images into a network input.

	Assumes images are already prepared (means subtracted, BGR order, ...).
	"""
	max_shape = np.array([im.shape for im in ims]).max(axis=0)
	num_images = len(ims)
	blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
					dtype=np.float32)
	for i in xrange(num_images):
		im = ims[i]
		blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
	# Move channels (axis 3) to axis 1
	# Axis order will become: (batch elem, channel, height, width)
	channel_swap = (0, 3, 1, 2)
	blob = blob.transpose(channel_swap)
	return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
	"""Mean subtract and scale an image for use in a blob."""
	im = im.astype(np.float32, copy=False)
	im -= pixel_means
	im_shape = im.shape
	im_size_min = np.min(im_shape[0:2])
	im_size_max = np.max(im_shape[0:2])
	im_scale = float(target_size) / float(im_size_min)
	# Prevent the biggest axis from being more than MAX_SIZE
	if np.round(im_scale * im_size_max) > max_size:
		im_scale = float(max_size) / float(im_size_max)
	im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
					interpolation=cv2.INTER_LINEAR)

	return im, im_scale
