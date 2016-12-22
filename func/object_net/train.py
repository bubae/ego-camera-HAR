import numpy as np
import os, sys
import caffe
from object_net.config import cfg as net_cfg
from caffe.proto import caffe_pb2
import google.protobuf as pb2
from utils.timer import Timer
from sendMail import send_mail

class SolverWrapper(object):
	def __init__(self, solver_prototxt, db, test_id, pretrained_model=None, width=80):
		# self.output_dir = os.path.join(db.dstDir, 'models');
		self.output_dir = os.path.join(db.dstDir, 'snapshot', 'object_net');

		self._test_id = test_id;

		self.solver = caffe.SGDSolver(solver_prototxt)
		if pretrained_model is not None:
			print ('Loading pretrained model '
				   'weights from {:s}').format(pretrained_model)
			self.solver.net.copy_from(pretrained_model)

		self.solver_param = caffe_pb2.SolverParameter();
		with open(solver_prototxt, 'rt') as f:
			pb2.text_format.Merge(f.read(), self.solver_param)

		self.solver.net.layers[0].set_db(db, test_id, width);

		# interp_layers = [k for k in self.solver.net.params.keys() if 'up' in k]
		# surgery.interp(self.solver.net, interp_layers)

	def snapshot(self):
		net = self.solver.net

		if not os.path.exists(self.output_dir):
			os.makedirs(self.output_dir)

		filename = (self.solver_param.snapshot_prefix + '_iter_%d_%d' % (self._test_id, self.solver.iter) + '.caffemodel')

		filepath = os.path.join(self.output_dir, filename)

		net.save(str(filepath))
		print 'Wrote snapshot to: {:s}'.format(filepath)

		print "snapshot"

	def train_model(self, max_iters):
		last_snapshot_iter = -1
		timer = Timer();

		while self.solver.iter < max_iters:
			timer.tic();
			self.solver.step(1)
			timer.toc();			

			if self.solver.iter % (10 * self.solver_param.display) == 0:
				print '%d Task %d iteration' % (self._test_id, self.solver.iter)
				print 'speed: {:.3f}s / iter'.format(timer.average_time)

			if self.solver.iter % net_cfg.TRAIN.SNAPSHOT_ITERS == 0:
				last_snapshot_iter = self.solver.iter
				self.snapshot();


		if last_snapshot_iter != self.solver.iter:
			self.snapshot();

def train_net(solver_prototxt, db, testset_id, pretrained_model=None, max_iters=40000, width=80):
	# "Train Object Network"

	sw = SolverWrapper(solver_prototxt, db, testset_id, pretrained_model, width);

	print 'Solving...'
	sw.train_model(max_iters)
	print 'Done solving'

	send_mail("widnanpear@gmail.com", "widianpear@kaist.ac.kr", [], "Train Finished %d" % width, "Train Finished %d" % width, None);




# class SolverWrapper(object):

# 	def __init__(self, solver_prototxt, db, pretrained_model=None):
# 		self.output_dir = db.dstDir
# 		print self.output_dir

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

# 		infix = ('_' + net_config.TRAIN.SNAPSHOT_INFIX if cfg.TRAIN.SNAPSHOT_INFIX != ' ' else '')
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
