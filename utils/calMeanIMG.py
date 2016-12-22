import numpy as np 
import cv2
import os, sys

def calMeanIMG(db):
    numVideo = len(db.vid2name);
    startIdx = 0;
    startFrame = 0;

    frameDir = os.path.join('/home/bbu/Workspace/data/result/%s' % db.name, 'CROP_IMAGE');


    im_means = np.zeros(3);

    for i in xrange(len(db.iid2path)):
        im = cv2.imread(os.path.join(frameDir, '%d.jpg' % i));

        im_means = im_means + np.mean(np.mean(im, axis=0), axis=0);

        print '%d / %d images processed' % (i, len(db.iid2path));


    im_means = im_means / len(db.iid2path);

    print im_means
        
    return [[im_means]];


def calMeanIMGFlow(db):
    numVideo = len(db.vid2name);
    startIdx = 0;
    startFrame = 0;

    frameDir = os.path.join('/home/bbu/Workspace/data/result/%s' % db.name, 'denseflow');

    imList = os.listdir(frameDir);

    im_means = np.zeros(3);

    cnt = 0;
    for file in imList:
        im = cv2.imread(os.path.join(frameDir, file));
        im_means = im_means + np.mean(np.mean(im, axis=0), axis=0);

        print '%d / %d images processed' % (cnt, len(imList));
        cnt +=1;
        
    im_means = im_means / len(imList);

    # for i in xrange(len(db.iid2path)):
    #     im = cv2.imread(os.path.join(frameDir, '%d.jpg' % i));

    #     im_means = im_means + np.mean(np.mean(im, axis=0), axis=0);

    #     print '%d / %d images processed' % (i, len(db.iid2path));


    # im_means = im_means / len(db.iid2path);
    
    # return [[im_means]];

    print im_means    
    # for i in xrange(len(db.iid2path)):
    #     im = cv2.imread(db.iid2path[i]);

    #     im_means = im_means + np.mean(np.mean(im, axis=0), axis=0);

    #     print '%d / %d images processed' % (i, len(db.iid2path));

    # im_means = im_means / len(db.iid2path);

    # print im_means;