import numpy as np
import os
import cv2

def videoToFrame(videoPath, desDirPath, frameRate):
	if not os.path.exists(desDirPath):
		os.makedirs(desDirPath);

	print videoPath
	cap = cv2.VideoCapture(videoPath);

	# print cap.isOpened();	

	frameLength = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT);
	# frameRate = cap.get(cv2.cv.CV_CAP_PROP_FPS);
	frameDownsampling = 1;

	# print cap.isOpened(), frameLength, frameRate
	cnt = 0;
	while(cap.isOpened()):
		ret, image = cap.read();

		if ret == 0:
			break;

		savePath = os.path.join(desDirPath, '%06d.jpg' % cnt);
		cv2.imwrite(savePath, image);

		cnt = cnt + 1;

		if frameLength == cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES):
			break;

		# print image.shape
		# break;


	# 	savePath = os.path.join(desDirPath, '');

	cap.release();



def main():
	# for dataset in dataSetList:
	# 	imdb = DataSet(dataset);
	# 	imdb.loadVideoList();

	homeDirs = ['/home/bbu/Workspace/data/GTEA_GAZE', '/home/bbu/Workspace/data/GTEA_GAZE_PLUS']
	# homeDirs = ['/home/bbu/Workspace/data/GTEA_GAZE_PLUS']
	extension = ['.mpg', '.avi']
	frameRate = [30.0, 24.0]

	for idx, homeDir in enumerate(homeDirs):
		videoDir = os.path.join(homeDir, 'Videos');
		frameDir = os.path.join(homeDir, 'frames');


		fileList = os.listdir(videoDir);

		for file in fileList:
			filePath = os.path.join(videoDir, file);

			if os.path.isdir(filePath):
				subfileList = os.listdir(filePath);

				for video in subfileList:
					dstDirPath = os.path.join(frameDir, file, video[:-4])
					videoPath = os.path.join(filePath, video);
					print videoPath
					videoToFrame(videoPath, dstDirPath, frameRate[idx]);

			else:
				dstDirPath = os.path.join(frameDir, file[:-4]);
				print filePath
				videoToFrame(filePath, dstDirPath, frameRate[idx]);

if __name__== "__main__":
	main();