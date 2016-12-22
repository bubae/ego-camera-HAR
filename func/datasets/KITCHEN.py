import os.path as osp
import sys
import numpy as np
from os import listdir
from easydict import EasyDict as edict
import cv2

np.set_printoptions(threshold='nan')

def label_parser(labelPath, mcid2name, ocid2name, acid2name):

	f = open(labelPath, "r");

	motion_labels = []
	object_labels = []
	activity_labels = []

	frameIndex = np.array([]);
	iid2xid = np.array([]);

	xid = 0;
	while True:
		line = f.readline();

		if not line:
			break;

		if len(line.split(' ')) < 3:
			continue;			

		startFrame = int(line.split(' ')[0]) - 1;
		endFrame = int(line.split(' ')[1]) - 1;

		label = line.split(' ')[2];

		motion_label = label.split('-')[0];
		object_label1 = label.split('-')[1];
		object_label2 = label.split('-')[3][:-1];

		if len(object_label2) > 0:
			if len(object_label1) == 0:
				object_label = object_label2;
			else:
				object_label = '_'.join([object_label1, object_label2]);
		else:
			object_label = object_label1;


		if motion_label == 'none' or len(object_label) == 0: 
			continue;

		activity_label = '_'.join([motion_label, object_label]);

		mcid = np.where(mcid2name == motion_label)[0]
		ocid = np.where(ocid2name == object_label)[0]
		acid = np.where(acid2name == activity_label)[0]

		if len(acid) == 0:
			continue;

		mcid = mcid[0];
		ocid = ocid[0];
		acid = acid[0];

		motion_labels.append(mcid);
		object_labels.append(ocid);
		activity_labels.append(acid);

		frameIndex = np.concatenate((frameIndex,range(startFrame, endFrame + 1)));
		iid2xid = np.concatenate((iid2xid, np.ones(endFrame - startFrame + 1) * xid));
		xid = xid +1;

		# print line, mcid, ocid, acid, activity_label

	f.close();
	return motion_labels, object_labels, activity_labels, frameIndex, iid2xid

def DB_KITCHEN(root):
	print "DB_KITCHEN";

	video_extension = '.avi'

	videoDirPath = osp.join(root, 'Videos');
	labelDirPath = osp.join(root, 'labels');
	handmaskRoot = osp.join(root, 'HandMask');
	handmaskGT   = osp.join(handmaskRoot, 'GroundTruth');
	handmaskIM   = osp.join(handmaskRoot, 'Images');
	frameRoot    = osp.join(root, 'frames');

	db = edict();
	db.opts = edict();
	db.vid2path = np.array([]); #
	db.vid2sid = np.array([]); #
	db.vid2name = np.array([]); #
	db.ocid2name = np.array([]);
	db.mcid2name = np.array([]);
	db.acid2name = np.array([]);
	db.iid2path = np.array([]);
	db.iid2xid = np.array([]);
	db.xid2mcid = np.array([]);
	db.xid2ocid = np.array([]);
	db.xid2acid = np.array([]);
	db.xid2sid = np.array([]);

	vid2name = [f.replace(video_extension, '') for f in listdir(videoDirPath) if osp.isfile(osp.join(videoDirPath, f))];
	vid2sid = range(len(vid2name));
	vid2path = np.array([osp.join(videoDirPath, f + video_extension) for f in vid2name]);

	mtnClassListPath = open(osp.join(root, 'motion_label.txt'), 'r');
	objClassListPath = open(osp.join(root, 'object_label.txt'), 'r');
	actClassListPath = open(osp.join(root, 'activity_label.txt'), 'r');

	mcid2name = np.array(mtnClassListPath.read().split('\n')[:-1]);
	ocid2name = np.array(objClassListPath.read().split('\n')[:-1]);
	acid2name = np.array(actClassListPath.read().split('\n')[:-1]);

	xid2mcid = np.array([]);
	xid2ocid = np.array([]);
	xid2acid = np.array([]);
	xid2sid = np.array([]);
	iid2path = np.array([]);
	iid2xid = np.array([]);

	for idx in xrange(len(vid2name)):
		vname = vid2name[idx];
		sid = vid2sid[idx];
		frameDirPath = osp.join(frameRoot, vname);
		labelPath = osp.join(labelDirPath, vname, 'labels.dat');
		motion_labels, object_labels, activity_labels, frameIndex, sub_iid2xid = label_parser(labelPath, mcid2name, ocid2name, acid2name);

		xid2mcid = np.concatenate((xid2mcid, motion_labels));
		xid2ocid = np.concatenate((xid2ocid, object_labels));
		xid2acid = np.concatenate((xid2acid, activity_labels));
		xid2sid = np.concatenate((xid2sid, np.ones(len(motion_labels))*sid))

		framePaths = [osp.join(frameDirPath, "frame%06d.jpg" % i) for i in frameIndex];

		iid2path = np.concatenate((iid2path, framePaths))
		# print vname, idx
		if idx == 0:
			iid2xid = np.array(sub_iid2xid);
		else:
			# print iid2xid
			iid2xid = np.concatenate((iid2xid, sub_iid2xid + iid2xid[-1] + 1));

	# print mcid2name, ocid2name, acid2name

	db.vid2name = vid2name;	
	db.vid2sid = vid2sid;
	db.vid2path = vid2path;
	db.mcid2name = mcid2name;
	db.ocid2name = ocid2name;
	db.acid2name = acid2name;
	db.iid2path = iid2path;
	db.iid2xid = iid2xid;
	db.xid2mcid = xid2mcid;
	db.xid2ocid = xid2ocid;
	db.xid2acid = xid2acid;
	db.xid2sid = xid2sid;

	return db
	# 
# DB_KITCHEN('/home/bbu/Workspace/data/KITCHEN')