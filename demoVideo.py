import init_path
import numpy as np
import os, sys
import cv2
from Db import Db
from init_path import cfg
from easydict import EasyDict as edict
from sendMail import send_mail


if __name__ == "__main__":
	setting = edict();
	setting.gpu 			= 1;
	setting.db 				= cfg.db.gtea;

	db = Db(setting.db, cfg.dstDir);
	db.genDb();

	labelPath = 'data/result/demo/video_label.txt';

	labels, frameIndex = np.loadtxt(labelPath, delimiter=',');

	videoPath = 'data/demo_videos/S4_Coffee_C1.mp4';

	cap = cv2.VideoCapture(videoPath);

	fourcc = cv2.cv.CV_FOURCC('M', 'J', 'P', 'G')
	out = cv2.VideoWriter('output.avi', fourcc, 15.0, (720,404))

	predicts = np.ones(964) * -1;
	# print np.ones(964) * -1;

	# iids = [];
	# for i in xrange(len(db.iid2xid)):
	# 	if np.char.find(db.iid2path[i], "S4_Coffee_C1") >= 0:
	# 		print db.iid2path[i]
	# 		iids.append(i);

	# start = iids[0];
	# baseFrame = 15;

	# frameIndex = frameIndex - start + baseFrame;

	frameIndex = [ int(db.iid2path[x][-8:-4]) for x in frameIndex]

	for i in xrange(len(frameIndex)):
		predicts[frameIndex[i]] = labels[i]
		# labels[i]
	# print frameIndex

	# for iid in frameIndex:
	# 	print int(db.iid2path[iid][-8:-4])

	# print frameIndex

	cnt = 0;

	print predicts
	# length = int(cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	# print length

	while(True):

		ret, frame = cap.read()

		if cnt > predicts.shape[0]:
			label_str = "None";
			label = -1;
		else:

			label = predicts[cnt];
			color = (255, 255, 255)
			label_str = "";

		if label == -1:
			label_str = "None";
			color = (0, 0, 255);
		else:
			label_str = np.str(db.acid2name[label]);
			label_list = label_str.split('_');
			color = (255, 0, 0);
			if len(label_list) >= 3:
				if label_list[2] == 'spoon':
					label_str = "%s %s with %s" % (label_list[0], label_list[1], label_list[2]);
				else:
					label_str = "%s %s in %s" % (label_list[0], label_list[1], label_list[2]);

			else:
				label_str = "%s %s" % (label_list[0], label_list[1])
		
		cnt = cnt +1;

		cv2.putText(frame, label_str, (10, 380), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2);

		out.write(frame);
		cv2.imshow('frame', frame);
		if cv2.waitKey(30) & 0xFF == ord('q'):
			break

	cap.release();
	out.release();
	cv2.destroyAllWindows();

	# print bb