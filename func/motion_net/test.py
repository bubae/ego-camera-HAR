import numpy as np
import os, sys
import caffe
from motion_net.config import cfg
from caffe.proto import caffe_pb2
import google.protobuf as pb2
from utils.timer import Timer
import cv2

def get_stackImg(iid, isFlip, db):


    ims = [];

    frameDir = '/home/bbu/Workspace/data/result/%s/denseflow' % db.name;

    for i in xrange(int(iid), int(iid)+cfg.TRAIN.NUM_STACK):

        im_x = cv2.imread(os.path.join(frameDir, 'flow_x_%05d.jpg' % i));
        im_x = im_x.astype(np.float32, copy=False)

        grey_im_x = cv2.cvtColor(im_x,cv2.COLOR_BGR2GRAY);
        # print grey_im_x.shape
        grey_im_x = cv2.resize(grey_im_x, (cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE));

        # print grey_im_x.shape
        if isFlip:
            grey_im_x = 255 - grey_im_x[:, ::-1]

        grey_im_x -= cfg.TRAIN.PIXEL_MEANS[0]
        ims.append(grey_im_x);

        im_y = cv2.imread(os.path.join(frameDir, 'flow_y_%05d.jpg' % i));
        im_y = im_y.astype(np.float32, copy=False)

        grey_im_y = cv2.cvtColor(im_y,cv2.COLOR_BGR2GRAY);

        grey_im_y = cv2.resize(grey_im_y, (cfg.TRAIN.MAX_SIZE, cfg.TRAIN.MAX_SIZE));

        grey_im_y -= cfg.TRAIN.PIXEL_MEANS[0]

        # cv2.imwrite('temp/%d_x.jpg' % i, grey_im_x);
        # cv2.imwrite('temp/%d_y.jpg' % i, grey_im_y);
        # print grey_im_x.shape

        if isFlip:
            grey_im_y = grey_im_y[:, ::-1]

        ims.append(grey_im_y);

    return np.array(ims);


def test_net(prototxtPath, modelPath, test_id, db):
    print 'Test Motion Networks'
    # sw = SolverWrapper(solver_prototxt, db, pretrained_model=pretrained_model)

    # vlen = [x.shape[0] for x in db.vid2albl];
    numBatch = 2;

    net = caffe.Net(prototxtPath, modelPath, caffe.TEST)
    net.name = 'test_motion_net'

    test_xids = np.where(db.xid2sid == test_id)[0];

    # if len(np.where(db.iid2xid == xid)[0]) >= 10
    # test_iids_list = np.array([np.where(db.iid2xid == xid)[0] for xid in test_xids if len(np.where(db.iid2xid == xid)[0]) >= 6]);



    dims = (numBatch, 2*cfg.TEST.NUM_STACK, 224, 224);

    cnt = 0;
    accuracy_score = 0;
    for xid in test_xids:
        iids = np.where(db.iid2xid == xid)[0];
        label = db.xid2mcid[xid];

        if len(iids) < cfg.TEST.NUM_STACK:
            continue

        cnt = cnt + 1;
        numData = len(iids) - cfg.TEST.NUM_STACK;

        # print len(iids), numData, range(numData)

        # print numData, label

        # processed_ims = [];

        scores_sum = np.zeros(10);

        for i in xrange(numData):

            processed_ims = [];

            stackImg = get_stackImg(iids[i], False, db)

            processed_ims.append(stackImg);

            flip_stackImg = get_stackImg(iids[i], True, db)

            processed_ims.append(flip_stackImg);

            im_blobs = np.zeros(shape=dims, dtype=np.float32)

            for j in xrange(numBatch):
                im_blobs[j,::] = processed_ims[j];

            # print im_blobs.shape
            blobs = {'data': im_blobs}

            blobs_out = net.forward(**blobs);

            scores = net.blobs['fc8'].data.copy();
            # print iids[i], scores
            scores_sum = scores_sum + scores.sum(axis=0);

            # print scores_sum
        # scores_sum = scores_sum / (numData*2);

        print "Task %d: %d / %d" % (xid, cnt, len(test_xids))
        # print label, scores_sum
        # print label, np.where(scores_sum==scores_sum.max())[0][0]

        predict = np.where(scores_sum==scores_sum.max())[0][0];

        if int(label) == int(predict):
            accuracy_score +=1;


    accuracy = np.float32(accuracy_score) / np.float32(cnt);

    return accuracy


def extract_motion_feature(prototxtPath, modelPath, db, testid):
    # prototxtPath = 'models/%s/objectnet/VGG_CNN_M_2048/deploy.prototxt' % db.name
    # modelPath = 'data/snapshot/vgg_cnn_m_2048_iter_4_10000.caffemodel';

    net = caffe.Net(prototxtPath, modelPath, caffe.TEST)
    net.name = 'test_motion_net'

    numBatch = 2;

    dims = (numBatch, 2*cfg.TRAIN.NUM_STACK, 224, 224);

    xids = range(int(db.iid2xid.max())+1);

    iid_lists = [np.where(db.iid2xid == xid)[0][:-(cfg.TRAIN.NUM_STACK)] for xid in xids];


    iids = np.array([]);

    fc6_dstDir = os.path.join(db.dstDir, 'Motionfeature', 'S_%d_%d' % (testid, 10), 'fc6');
    fc7_dstDir = os.path.join(db.dstDir, 'Motionfeature', 'S_%d_%d' % (testid, 10), 'fc7');

    if not os.path.exists(fc6_dstDir):
        os.makedirs(fc6_dstDir);    

    if not os.path.exists(fc7_dstDir):
        os.makedirs(fc7_dstDir);    

    for iid_list in iid_lists:
        iids = np.concatenate((iids, iid_list));

    cnt = 0;

    print iids.shape, iids.max()

    for iid in iids:
        processed_ims = [];

        stackImg = get_stackImg(iid, False, db)

        processed_ims.append(stackImg);

        flip_stackImg = get_stackImg(iid, True, db)

        processed_ims.append(flip_stackImg);


        im_blobs = np.zeros(shape=dims, dtype=np.float32)

        for i in xrange(numBatch):
            im_blobs[i,::] = processed_ims[i];

        # print im_blobs.shape
        blobs = {'data': im_blobs}

        blobs_out = net.forward(**blobs);

        fc6_feature = net.blobs['fc6'].data.copy();
        fc7_feature = net.blobs['fc7'].data.copy();

        fc6_featurePath = os.path.join(fc6_dstDir, '%d.npy' % (iid + int(cfg.TRAIN.NUM_STACK/2)));
        np.save(fc6_featurePath, fc6_feature[0]);

        fc7_featurePath = os.path.join(fc7_dstDir, '%d.npy' % (iid + int(cfg.TRAIN.NUM_STACK/2)));
        np.save(fc7_featurePath, fc7_feature[0]);


        fc6_featurePath = os.path.join(fc6_dstDir, '%d_flip.npy' % (iid + int(cfg.TRAIN.NUM_STACK/2)));
        np.save(fc6_featurePath, fc6_feature[1]);

        fc7_featurePath = os.path.join(fc7_dstDir, '%d_flip.npy' % (iid + int(cfg.TRAIN.NUM_STACK/2)));
        np.save(fc7_featurePath, fc7_feature[1]);

        print "%d / %d" % (cnt, len(iids));
        cnt = cnt + 1;


def get_y_predict(prototxtPath, modelPath, db, testid):
    # prototxtPath = 'models/%s/objectnet/VGG_CNN_M_2048/deploy.prototxt' % db.name
    # modelPath = 'data/snapshot/vgg_cnn_m_2048_iter_4_10000.caffemodel';

    net = caffe.Net(prototxtPath, modelPath, caffe.TEST)
    net.name = 'test_motion_net'

    numBatch = 1;

    dims = (numBatch, 2*cfg.TRAIN.NUM_STACK, 224, 224);

    xids = range(int(db.iid2xid.max())+1);

    iid_lists = [np.where(db.iid2xid == xid)[0][:-(cfg.TRAIN.NUM_STACK)] for xid in xids];

    print dims

    iids = np.array([]);

    # fc6_dstDir = os.path.join(db.dstDir, 'Motionfeature', 'S_%d_%d' % (testid, 10), 'fc6');
    # fc7_dstDir = os.path.join(db.dstDir, 'Motionfeature', 'S_%d_%d' % (testid, 10), 'fc7');

    # if not os.path.exists(fc6_dstDir):
    #     os.makedirs(fc6_dstDir);    

    # if not os.path.exists(fc7_dstDir):
    #     os.makedirs(fc7_dstDir);    

    y_predict = os.path.join(db.dstDir, 'Motionfeature', 'y_predict_%d_%d' % (testid, 10));

    if not os.path.exists(y_predict):
        os.makedirs(y_predict);    


    for iid_list in iid_lists:
        iids = np.concatenate((iids, iid_list));

    cnt = 0;

    print iids.shape, iids.max()
    iids = iids.astype(int);
    y_target = np.array(db.xid2mcid[db.iid2xid[iids].astype(int)].astype(int))
    y_predicts = np.array([]);

    # print y_target

    savePath = os.path.join('data/result/confusion_matrix/motion/predict_result_%d.npy' % testid);

    for idx in xrange(0,len(iids),numBatch):
        processed_ims = [];

        # if idx + len
        for i in xrange(numBatch):
            stackImg = get_stackImg(iids[idx+i], False, db)

            processed_ims.append(stackImg);

            # flip_stackImg = get_stackImg(iids[idx+i], True, db)

            # processed_ims.append(flip_stackImg);

        im_blobs = np.zeros(shape=dims, dtype=np.float32)

        for i in xrange(numBatch):
            im_blobs[i,::] = processed_ims[i];

        print im_blobs.shape
        # print im_blobs.shape
        blobs = {'data': im_blobs}

        blobs_out = net.forward(**blobs);

        y_score = net.blobs['cls_prob'].data.copy();
        # fc7_feature = net.blobs['fc7'].data.copy();

        y_predict = np.argmax(y_score, axis=1)

        y_predicts = np.concatenate([y_predicts, y_predict]);

        print y_predicts.shape
        # fc6_featurePath = os.path.join(fc6_dstDir, '%d.npy' % (iid + int(cfg.TRAIN.NUM_STACK/2)));
        # np.save(fc6_featurePath, fc6_feature[0]);

        # fc7_featurePath = os.path.join(fc7_dstDir, '%d.npy' % (iid + int(cfg.TRAIN.NUM_STACK/2)));
        # np.save(fc7_featurePath, fc7_feature[0]);


        # fc6_featurePath = os.path.join(fc6_dstDir, '%d_flip.npy' % (iid + int(cfg.TRAIN.NUM_STACK/2)));
        # np.save(fc6_featurePath, fc6_feature[1]);

        # fc7_featurePath = os.path.join(fc7_dstDir, '%d_flip.npy' % (iid + int(cfg.TRAIN.NUM_STACK/2)));
        # np.save(fc7_featurePath, fc7_feature[1]);

        cnt = cnt + numBatch;

        print "%d / %d" % (cnt, len(iids));

    y_predicts = np.array(y_predicts);
    np.save(savePath, np.array([y_target, y_predicts]));
    # print xids, len(xids)
    # iids_list = np.array([np.where(db.iid2xid == xid)[0] for xid in xids if len(np.where(db.iid2xid == xid)[0]) >= 11]);

    # dims = (numBatch, 2*cfg.TRAIN.NUM_STACK, 224, 224);

    # for xid in xids:
    #     iids = np.where(db.iid2xid == xid)[0];
    #     label = db.xid2mcid[xid];

    #     if len(iids) < cfg.TEST.NUM_STACK:
    #         continue

    #     numData = len(iids) - cfg.TEST.NUM_STACK;

    # prototxtPath = 'models/%s/motionnet/test.prototxt' % db.name;
    # modelPath = 'data/motion_net/vgg_16_action_flow_pretrain.caffemodel';

    # net = caffe.Net(prototxtPath, modelPath, caffe.TEST)
    # net.name = 'test_object_net'

    # frameDir = '/home/bbu/Workspace/data/result/%s/denseflow' % db.name;

    # fc6_dstDir = os.path.join(db.dstDir, 'Motionfeature', 'fc6');
    # fc7_dstDir = os.path.join(db.dstDir, 'Motionfeature', 'fc7');

    # if not os.path.exists(fc6_dstDir):
    #     os.makedirs(fc6_dstDir);    

    # if not os.path.exists(fc7_dstDir):
    #     os.makedirs(fc7_dstDir);

    # train_task_ids = np.where(db.xid2sid != 4)[0];
    # train_iid_list = [np.where(db.iid2xid == xid)[0][:-(cfg.TRAIN.NUM_STACK)] for xid in train_task_ids];

    # train_iids = np.array([]);
    # for iid_list in train_iid_list:
    #     train_iids = np.concatenate((train_iids, iid_list));

    # for idx in xrange(len(train_iids)):
    #     print idx, train_iids[idx];
    #     iid = train_iids[idx];

    #     blob = get_minibatch([iid], [0], db);

    #     print blob['data'].shape
    #     blobs_out = net.forward(**blob);

    #     fc6_feature = net.blobs['fc6'].data.copy();
    #     fc6_featurePath = os.path.join(fc6_dstDir, '%d.npy' % iid);

    #     fc7_feature = net.blobs['fc7'].data.copy();
    #     fc7_featurePath = os.path.join(fc7_dstDir, '%d.npy' % iid);

    #     print 'Feature Extract %d / %d Feature Saved' % (iid, len(train_iids));
    #     np.save(fc6_featurePath, fc6_feature);
    #     np.save(fc7_featurePath, fc7_feature);