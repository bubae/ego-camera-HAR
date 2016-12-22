import init_path
import caffe
from caffe.proto import caffe_pb2
import google.protobuf as pb2
import numpy as np
import os, sys, cv2
from easydict import EasyDict as edict
from utils.timer import Timer
from utils.sendMail import send_mail
from config import cfg
import surgery, score
import setproctitle

class SolverWrapper(object):
	def __init__(self, solver_prototxt, db, pretrained_model=None):
		# self.output_dir = os.path.join(db.dstDir, 'models');

		self.solver = caffe.SGDSolver(solver_prototxt)
		if pretrained_model is not None:
			print ('Loading pretrained model '
				   'weights from {:s}').format(pretrained_model)
			self.solver.net.copy_from(pretrained_model)

		self.solver_param = caffe_pb2.SolverParameter();
		with open(solver_prototxt, 'rt') as f:
			pb2.text_format.Merge(f.read(), self.solver_param)

		self.solver.net.layers[0].set_db(db);

		interp_layers = [k for k in self.solver.net.params.keys() if 'up' in k]
		surgery.interp(self.solver.net, interp_layers)


	def snapshot(self):
		net = self.solver.net
		print "snapshot"

	def train_model(self, max_iters):
		last_snapshot_iter = -1
		timer = Timer();

		while self.solver.iter < max_iters:
			timer.tic();
			self.solver.step(1)
			timer.toc();			

			if self.solver.iter % (10 * self.solver_param.display) == 0:
				print 'speed: {:.3f}s / iter'.format(timer.average_time)

			if self.solver.iter % cfg.TRAIN.SNAPSHOT_ITERS == 0:
				last_snapshot_iter = self.solver.iter
				self.snapshot();
		# if last_snapshot_iter != self.solver.iter:
		# 	model_paths.append(self.snapshot())

def get_db():

	labelDirPath = 'data/HandMask/GroundTruth'
	imageDirPath = 'data/HandMask/Images'

	imList = os.listdir(labelDirPath);

	db = [];
	for i in xrange(len(imList)):
		fileName = imList[i][:-4];
		imPath = os.path.join(imageDirPath, fileName + '.jpg');
		labelPath = os.path.join(labelDirPath, fileName + '.png');

		e = {'image': imPath, 'label': labelPath}
		db.append(e)

	return db

def train_net();

	db = get_db();
	solver_prototxt = 'models/GTEA/handsegmentenet/solver.prototxt';
	pretrained_model = 'data/hand_segmentation_net/GTEA_20000.caffemodel'
	max_iters = 20000;

	caffe.set_mode_gpu()
	caffe.set_device(0)

	sw = SolverWrapper(solver_prototxt, db, pretrained_model);

	print 'Solving...'
	sw.train_model(max_iters);
	print 'Done solving'

	send_mail("widnanpear@gmail.com", "widianpear@kaist.ac.kr", [], "Train Finished", "Train Finished", None);
