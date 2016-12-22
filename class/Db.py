# import init_path
import numpy as np
import pylab
import math
import os.path as osp
import os, sys
import pickle

class Db(object):
	def __init__(self, setting, dstDir):
		self.dstDir = osp.join(dstDir, setting.name);
		self.name = setting.name;
		self.funh = setting.funh;
		self.root = setting.root;
		self.opts = None;
		self.vid2name = np.array([]);
		self.vid2sid = np.array([]);
		self.vid2path = np.array([]); #
		self.mcid2name = np.array([]); #
		self.ocid2name = np.array([]); #
		self.acid2name = np.array([]);
		self.iid2path = np.array([]); #
		self.iid2xid = np.array([]); #
		self.xid2mcid = np.array([]); #
		self.xid2ocid = np.array([]); #
		self.xid2acid = np.array([]); #
		self.xid2sid = np.array([]); #

	def genDb(self):
		dbmodule = __import__(self.name);
		dbfunh = getattr(dbmodule, self.funh);

		fpath = self.getPath();
		try:
			print "%s: Try to load db." % os.path.basename(__file__);
			db = pickle.load(open(fpath));
			print "%s: db loaded." % os.path.basename(__file__);
		except:
			print "%s: Gen db." % os.path.basename(__file__);

			db = dbfunh(self.root);

			print "%s: Done." % os.path.basename(__file__);
			print "%s: Save DB." % os.path.basename(__file__);
			self.makeDir();
			pickle.dump(db, open(fpath, 'w'), pickle.HIGHEST_PROTOCOL);
			print "%s: Done." % os.path.basename(__file__);

		self.opts = db.opts;
		self.vid2name = db.vid2name;
		self.vid2sid = db.vid2sid;
		self.vid2path = db.vid2path; #
		self.mcid2name = db.mcid2name; #
		self.ocid2name = db.ocid2name; #
		self.acid2name = db.acid2name;
		self.iid2path = db.iid2path; #
		self.iid2xid = db.iid2xid; #
		self.xid2mcid = db.xid2mcid; #
		self.xid2ocid = db.xid2ocid; #
		self.xid2acid = db.xid2acid; #
		self.xid2sid = db.xid2sid; #

		return fpath

	def getName(self):
		name = 'DB';
		return name;

	def getDir(self):
		return self.dstDir;

	def makeDir(self):
		_dir = self.getDir();
		if not os.path.exists(_dir):
			os.mkdir(_dir);

	def getPath(self):
		fname = self.getName() + '.pkl';
		path = osp.join(self.getDir(), fname);
		return path;