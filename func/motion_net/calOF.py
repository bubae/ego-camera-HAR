import numpy as np
import os, sys
import cv2
from subprocess import call, check_output
# from motion_net.config import cfg as net_cfg
from config import cfg as net_cfg

def motionRemove(prevImg, nextImg):
    hsv = np.zeros_like(nextImg)
    hsv[...,1] = 255

    surf = cv2.SURF(net_cfg.OPT_FLOW.SURF_THRESHOLD);
    kp1, des1 = surf.detectAndCompute(prevImg,None);    
    kp2, des2 = surf.detectAndCompute(nextImg, None); 

    if len(kp1) < net_cfg.OPT_FLOW.KEYPOINT_THRESHOLD or len(kp1) < net_cfg.OPT_FLOW.KEYPOINT_THRESHOLD:
        return 0, 0;

    FLANN_INDEX_KDTREE = 0     
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1,des2,k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>10:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()

        h,w,c = prevImg.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
    else:
        return 0, 0;

    grey_prev_Img = cv2.cvtColor(prevImg,cv2.COLOR_RGB2GRAY);
    grey_next_Img = cv2.cvtColor(nextImg,cv2.COLOR_RGB2GRAY);

    perImg = cv2.warpPerspective(grey_prev_Img, M, grey_prev_Img.shape[::-1])

    PREV_B_IMG = prevImg[:,:,0];
    PREV_G_IMG = prevImg[:,:,1];
    PREV_R_IMG = prevImg[:,:,2];

    perImg = cv2.warpPerspective(grey_prev_Img, M, grey_prev_Img.shape[::-1])

    R_IMG = cv2.warpPerspective(PREV_R_IMG, M, PREV_R_IMG.shape[::-1])
    G_IMG = cv2.warpPerspective(PREV_G_IMG, M, PREV_G_IMG.shape[::-1])
    B_IMG = cv2.warpPerspective(PREV_B_IMG, M, PREV_B_IMG.shape[::-1])

    HOMO_IMG = np.zeros(prevImg.shape, dtype='uint8')

    HOMO_IMG[:,:,0] = B_IMG
    HOMO_IMG[:,:,1] = G_IMG
    HOMO_IMG[:,:,2] = R_IMG

    ret, mask = cv2.threshold(perImg, 5, 255, cv2.THRESH_BINARY)

    masking_img = cv2.bitwise_and(nextImg, nextImg, mask = mask)

    return HOMO_IMG, masking_img

def calFlowFarneback(prevImg, Img):
    
    prev_grey = cv2.cvtColor(prevImg,cv2.COLOR_BGR2GRAY);
    grey = cv2.cvtColor(Img,cv2.COLOR_BGR2GRAY);


    flow = cv2.calcOpticalFlowFarneback(prev_grey, grey, 0.702, 5, 10, 2, 7, 1.5, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)

    out_x = np.zeros_like(flow[:,:,0]);
    out_y = np.zeros_like(flow[:,:,0]);

    # print out.shape
    for i in xrange(flow.shape[0]):
        for j in xrange(flow.shape[1]):
            x = flow[i,j,0];
            y = flow[i,j,1];
            out_x[i,j] = CAST(x, -20, 20);
            out_y[i,j] = CAST(y, -20, 20);


    print flow.shape

    # temp1 = np.concatenate((prevImg, Img)) 
    # temp = np.concatenate((flow[:,:,0], flow[:,:,1]))
    # cv2.imshow('frame', temp1);
    # cv2.waitKey(0)


def motion_net(db, setting):


    f = open('data/motion_net/OF_LIST.txt','w');

    xlen = len(db.xid2sid);

    callMtd = 'lib/dense_flow/denseImage_gpu';

    # print db.iid2path[iids[0]]

    dstPath = os.path.join(db.dstDir, "denseflow_backmotion");

    if not os.path.exists(dstPath):
        os.makedirs(dstPath);

    for xid in xrange(xlen):
        iids = np.where(db.iid2xid == xid)[0]

        for idx in xrange(len(iids) - 1):

            line = "%s %s %d\n" % (db.iid2path[iids[idx]], db.iid2path[iids[idx+1]], iids[idx]);
            f.write(line)
            # if os.path.exists(os.path.join(dstPath, "flow_x_%05d.jpg" % iids[idx])):
            #     print "Skip %d" % iids[idx]
            #     continue;

            # prevImg = cv2.imread(db.iid2path[iids[idx]]);
            # nextImg = cv2.imread(db.iid2path[iids[idx+1]]);
            
            # print db.iid2path[iids[idx]], db.iid2path[iids[idx+1]]

            # out1, out2 = motionRemove(prevImg, nextImg);

            # prevPath = os.path.join(db.dstDir, 'flow_prev.jpg');
            # nextPath = os.path.join(db.dstDir, 'flow_next.jpg');

            # cv2.imshow('frame1', out1);
            # cv2.imshow('frame2', out2);
            # cv2.waitKey(20);
            # cv2.imwrite(prevPath, out1);
            # cv2.imwrite(nextPath, out2);

            # print "%d / %d: %d / %d: %d" % (xid, xlen, idx, len(iids), iids[idx])

            # call([callMtd, "-p", db.iid2path[iids[idx]], "-i", db.iid2path[iids[idx+1]], "-x", os.path.join(dstPath, "flow_x"), "-y", os.path.join(dstPath, "flow_y"), "-d", "0", "-n", str(iids[idx])])


    f.close();

# prev_img = '/home/bbu/Workspace/working/ego-camera-HAR/lib/dense_flow/11.jpg'
# prevImg = cv2.imread(prev_img)
# next_img = '/home/bbu/Workspace/working/ego-camera-HAR/lib/dense_flow/22.jpg'
# nextImg = cv2.imread(next_img)

# aa, bb = motionRemove(prevImg, nextImg);

# cv2.imwrite('HOMO_IMG.jpg', aa);
# cv2.imwrite('MASK_IMG.jpg', bb);