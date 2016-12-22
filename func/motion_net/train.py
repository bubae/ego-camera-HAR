import numpy as np
import os, sys
import caffe
from motion_net.config import cfg as net_cfg
from caffe.proto import caffe_pb2
import google.protobuf as pb2
from utils.timer import Timer

# class SolverWrapper(object):

# 	def __init__(self, solver_prototxt, db, pretrained_model=None):
# 		self.output_dir = os.path.join(db.dstDir, 'models');

# 		self.solver = caffe.SGDSolver(solver_prototxt)
# 		if pretrained_model is not None:
# 			print ('Loading pretrained model '
# 				   'weights from {:s}').format(pretrained_model)
# 			self.solver.net.copy_from(pretrained_model)

# 		self.solver_param = caffe_pb2.SolverParameter();
# 		with open(solver_prototxt, 'rt') as f:
# 			pb2.text_format.Merge(f.read(), self.solver_param)

# 		self.solver.net.layers[0].set_db(db);

# 	def snapshot(self):
# 		net = self.solver.net

# 		if not os.path.exists(self.output_dir):
# 			os.makedirs(self.output_dir)

# 		infix = ('_' + net_config.TRAIN.SNAPSHOT_INFIX if net_config.TRAIN.SNAPSHOT_INFIX != ' ' else '')
# 		filename = (self.solver_param.snapshot_prefix + infix + '_iter_{:d}'.format(self.solver.iter) + '.caffemodel')
# 		filename = os.path.join(self.output_dir, filename)

# 		net.save(str(filename))
# 		print 'Wrote snapshot to: {:s}'.format(filename)

# 		return filename

# 	def train_model(self, max_iters):
# 		# print "hihi"
# 		last_snapshot_iter = -1
# 		timer = Timer();
# 		model_paths = []
# 		while self.solver.iter < max_iters:
# 			timer.tic();
# 			self.solver.step(1)
# 			timer.toc();

# 			if self.solver.iter % (10 * self.solver_param.display) == 0:
# 				print 'speed: {:.3f}s / iter'.format(timer.average_time)

# 			if self.solver.iter % net_config.TRAIN.SNAPSHOT_ITERS == 0:
# 				last_snapshot_iter = self.solver.iter
# 				model_paths.append(self.snapshot())

# 		if last_snapshot_iter != self.solver.iter:
# 			model_paths.append(self.snapshot())

# 		return model_paths

# def train_net(solver_prototxt, db, pretrained_model=None, max_iters=40000):
# 	# "Train Object Network"

# 	sw = SolverWrapper(solver_prototxt, db, pretrained_model=pretrained_model)

# 	print 'Solving...'
# 	model_paths = sw.train_model(max_iters)
# 	print 'Done solving'
# 	# return model_paths





class SolverWrapper(object):

	def __init__(self, solver_prototxt, db, test_id, pretrained_model=None, startIter=0):
		self.output_dir = os.path.join(db.dstDir, 'snapshot', 'motion_net');

		self._test_id = test_id;
		self._startIter = startIter;

		self.solver = caffe.SGDSolver(solver_prototxt)
		if pretrained_model is not None:
			print ('Loading pretrained model '
				   'weights from {:s}').format(pretrained_model)
			self.solver.net.copy_from(pretrained_model)

		self.solver_param = caffe_pb2.SolverParameter();
		with open(solver_prototxt, 'rt') as f:
			pb2.text_format.Merge(f.read(), self.solver_param)

		self.solver.net.layers[0].set_db(db, test_id);

		# aa = [net.params[k][0].data.shape for k in self.solver.net.params.keys()]
		print [self.solver.net.params[k][0].data.shape for k in self.solver.net.params.keys()]

	def snapshot(self):
		net = self.solver.net

		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		filename = (self.solver_param.snapshot_prefix + '_iter_%d_%d' % (self._test_id, self.solver.iter + self._startIter) + '.caffemodel')

		filepath = os.path.join(self.output_dir, filename)

		net.save(str(filepath))
		print 'Wrote snapshot to: {:s}'.format(filepath)

		print "snapshot"


	def train_model(self, max_iters):
		# print "hihi"
		last_snapshot_iter = -1
		timer = Timer();
		model_paths = []
		while (self._startIter + self.solver.iter) < max_iters:
			timer.tic();
			self.solver.step(1)
			timer.toc();
			if (self._startIter + self.solver.iter) % (self.solver_param.display) == 0:
				print "%d: %d iter Training..." % (self._test_id, self._startIter + self.solver.iter)

			if (self._startIter + self.solver.iter) % (10 * self.solver_param.display) == 0:
				print 'speed: {:.3f}s / iter'.format(timer.average_time)

			if (self._startIter + self.solver.iter) % net_cfg.TRAIN.SNAPSHOT_ITERS == 0:
				last_snapshot_iter = (self._startIter + self.solver.iter)
				model_paths.append(self.snapshot())

		if last_snapshot_iter != (self._startIter + self.solver.iter):
			model_paths.append(self.snapshot())

		return model_paths

def train_net(solver_prototxt, db, testset_id, pretrained_model=None, max_iters=30000, startIter=0):
	# "Train Object Network"

	print solver_prototxt
	sw = SolverWrapper(solver_prototxt, db, testset_id, pretrained_model, startIter)

	print 'Solving...'
	sw.train_model(max_iters)
	print 'Done solving'

	# send_mail("widnanpear@gmail.com", "widianpear@kaist.ac.kr", [], "Train Finished %d" % testset_id, "Train Finished %d" % testset_id, None);
