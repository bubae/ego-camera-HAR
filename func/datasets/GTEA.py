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
		line = f.readline()

		if not line:
			break;

		if len(line.split(' ')) < 3:
			continue;

		labels = line.split(' ')[0][1:-1];
		motion_label = labels.split('><')[0];
		object_label = '_'.join(labels.split('><')[1].split(','));
		activity_label = '_'.join([motion_label, object_label]);

		# print line[:-1]
		# print acid2name
		# print motion_label, object_label, activity_label, np.where(acid2name == activity_label)
		# print mcid, ocid, acid, mcid2name[mcid], ocid2name[ocid], acid2name[acid];

		mcid = np.where(mcid2name == motion_label)[0][0]
		ocid = np.where(ocid2name == object_label)[0][0]
		acid = np.where(acid2name == activity_label)[0][0]


		frameInter = line.split(' ')[1][1:-1];
		startFrame = int(frameInter.split('-')[0])
		endFrame = int(frameInter.split('-')[1])

		motion_labels.append(mcid)
		object_labels.append(ocid)
		activity_labels.append(acid)

		frameIndex = np.concatenate((frameIndex,range(startFrame, endFrame + 1)));
		iid2xid = np.concatenate((iid2xid, np.ones(endFrame - startFrame + 1) * xid));
		xid = xid +1;

	f.close();

	# print frameIndex
	# print iid2xid
	return motion_labels, object_labels, activity_labels, frameIndex, iid2xid

def DB_GTEA(root):
	print "DB_GTEA";

	video_extension = '.mp4'

	videoDirPath = osp.join(root, 'Videos');
	labelDirPath = osp.join(root, 'labels');
	handmaskRoot = osp.join(root, 'HandMask');
	handmaskGT   = osp.join(handmaskRoot, 'GroundTruth');
	handmaskIM   = osp.join(handmaskRoot, 'Images');
	frameRoot    = osp.join(root, 'frames');

	# vid = Video ID
	# ocid = Object class ID
	# mcid = Motion class ID
	# acid = Acitivty class ID
	# xid = Instance Class ID
	# sid = Subject ID
	# iid = Image(Frame) ID

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

	vid2name = [f.replace('.mp4', '') for f in listdir(videoDirPath) if osp.isfile(osp.join(videoDirPath, f))];
	vid2sid = [int(f[1])-1 for f in vid2name];
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
		# cap = cv2.VideoCapture(vid2path[idx]);
		# vlen = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

		# print vlen, vname
		frameDirPath = osp.join(frameRoot, vname);
		labelPath = osp.join(labelDirPath, vname + '.txt');

		motion_labels, object_labels, activity_labels, frameIndex, sub_iid2xid = label_parser(labelPath, mcid2name, ocid2name, acid2name);
		xid2mcid = np.concatenate((xid2mcid, motion_labels));
		xid2ocid = np.concatenate((xid2ocid, object_labels));
		xid2acid = np.concatenate((xid2acid, activity_labels));
		xid2sid = np.concatenate((xid2sid, np.ones(len(motion_labels))*sid))

		framePaths = [osp.join(frameDirPath, "frame%06d.jpg" % i) for i in frameIndex];

		# print framePaths
		iid2path = np.concatenate((iid2path, framePaths))
		# print vname, idx
		if idx == 0:
			iid2xid = np.array(sub_iid2xid);
		else:
			# print iid2xid
			iid2xid = np.concatenate((iid2xid, sub_iid2xid + iid2xid[-1] + 1));

	# print iid2xid
	# print acid2name[xid2acid[527]]

	# for path in iid2path:
	# 	if not osp.isfile(path):
	# 		print "no!!"
	# 		print path
	
	# print len(iid2path)
	# print len(iid2xid)
	# print xid2sid

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

	# print vid2path
	# db.opts = edict();
	# db.opts.handmask = 1;
	# db.opts.imframe = 1;
	# db.hid2gtpath = np.array([]); #
	# db.hid2impath = np.array([]); #
	# db.hid2vid = np.array([]); #
	# db.iid2path = np.array([]); #
	# db.iid2vid = np.array([]); #
	# db.ocid2name = np.array([]); #
	# db.mcid2name = np.array([]); #
	# db.vid2path = np.array([]); #
	# db.vid2lpath = np.array([]); #
	# db.vid2name = np.array([]); #
	# db.vid2cid = np.array([]); #
	# db.vid2len = np.array([]);
  
	# vid2vname = [f for f in listdir(videoDirPath) if osp.isfile(osp.join(videoDirPath, f))];
	# db.vid2path = np.array([osp.join(videoDirPath, f) for f in vid2vname]);
	# db.vid2name =  np.array([f.replace('.mp4', '') for f in vid2vname]);
	# db.vid2lpath = np.array([osp.join(labelDirPath, f.replace('.mp4', '.txt')) for f in vid2vname]);

	# hid2gtpath = [osp.join(handmaskGT, f) for f in listdir(handmaskGT) if osp.isfile(osp.join(handmaskGT, f))];
	# db.hid2gtpath = np.array(hid2gtpath);
	# hid2impath = [osp.join(handmaskIM, f) for f in listdir(handmaskIM) if osp.isfile(osp.join(handmaskIM, f))];
	# db.hid2impath = np.array(hid2impath);
	# hname = [osp.basename(f).split('_')[0:2] for f in db.hid2impath];
	# hname = np.array([ '_'.join([f[0].upper(), f[1].upper(), 'C1']) for f in hname]);

	# vid2nameUpper = np.array([f.upper() for f in db.vid2name]);
	# db.hid2vid = np.array([np.where(vid2nameUpper==vname)[0][0] for vname in hname]);

	# # vid2len = [];

	# # classLabel = open(ops.join(root, 'class_list.txt'), 'r');

	# for idx in xrange(len(db.vid2name)):
	# 	vname = db.vid2name[idx];
	# 	framePath = osp.join(frameRoot, vname);

	# 	cap = cv2.VideoCapture(db.vid2path[idx]);
	# 	vlen = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

	# 	iid2name
	# 	iid2path = np.array([osp.join])
	# 	db.vid2cid = np.concatenate((db.vid2cid, ));
	# 	db.vid2len = np.concatenate((db.vid2len, [vlen]));
	# 	db.iid2path = np.concatenate((db.iid2path, iid2path), axis=0);
	# 	db.iid2vid = np.concatenate((db.iid2vid, np.ones(iid2path.shape[0])*idx), axis=0)



	# db.vid2len = vid2len;
	# objClassLabel = open(osp.join(root, 'object_label.txt'), 'r');

	# db.ocid2name = np.array(['NONE'] + objClassLabel.read().split('\n')[:-1]);
	# actClassLabel = open(osp.join(root, 'action_label.txt'), 'r');
	# db.acid2name = np.array(['NONE'] + actClassLabel.read().split('\n')[:-1]);

	# db.vid2albl = [None] * len(db.vid2name); #
	# db.vid2olbl = [None] * len(db.vid2name); #


	# for idx in xrange(len(db.vid2name)):
	# 	vname = db.vid2name[idx];
	# 	framePath = osp.join(frameRoot, vname);

	# 	actLabel, objLabel = videoToLabelInfo(videoDirPath, labelDirPath, vname, db.ocid2name, db.acid2name);
	# 	db.vid2albl[idx] = actLabel;
	# 	db.vid2olbl[idx] = objLabel;

	# 	vlen = len(actLabel);
	# 	iid2name = np.array(['frame%06d.jpg'% x for x in range(vlen)]);
	# 	iid2path = np.array([osp.join(framePath, f) for f in iid2name if osp.isfile(osp.join(framePath, f))]);
	# 	db.iid2path = np.concatenate((db.iid2path, iid2path), axis=0);
	# 	db.iid2vid = np.concatenate((db.iid2vid, np.ones(iid2path.shape[0])*idx), axis=0);

	# for framePath in framePaths:
	# 	vname = osp.basename(framePath);
	# 	vid = np.where(db.vid2name==vname)[0][0];

	# 	actLabel, objLabel = videoToLabelInfo(videoDirPath, labelDirPath, vname, db.ocid2name, db.acid2name);
	# 	db.vid2albl[vid] = actLabel;
	# 	db.vid2olbl[vid] = objLabel;

		# iid2path = np.array([osp.join(framePath, f) for f in listdir(framePath) if osp.isfile(osp.join(framePath, f))]);
	# 	db.iid2path = np.concatenate((db.iid2path, iid2path), axis=0);
	# 	db.iid2vid = np.concatenate((db.iid2vid, np.ones(iid2path.shape[0])*vid), axis=0);
		# print db.iid2vid
		# print np.where(db.vid2name==vname)[0][0];
		# break;
	return db