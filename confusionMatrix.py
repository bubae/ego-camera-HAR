import init_path
import numpy as np
import os, sys
from Db import Db
from init_path import cfg
from easydict import EasyDict as edict
from sendMail import send_mail
from timer import Timer
import cv2
import itertools
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix


# import some data to play with


#   print array.shape
# iris = datasets.load_iris()
# X = iris.data
# y = iris.target
# class_names = iris.target_names

# print class_names
# # Split the data into a training set and a test set
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# # Run classifier, using a model that is too regularized (C too low) to see
# # the impact on the results
# classifier = svm.SVC(kernel='linear', C=0.01)
# y_pred = classifier.fit(X_train, y_train).predict(X_test)


# print y_pred.shape, y_test.shape

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.jet):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = cm*100
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')


    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, [], rotation=45)
    plt.yticks(tick_marks, classes)

    # print(cm)

    # thresh = cm.max() / 4.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, cm[i, j],
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')

# print class_names

# # Compute confusion matrix
# cnf_matrix = confusion_matrix(y_test, y_pred)
# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names,
#                       title='Confusion matrix, without normalization')

# # Plot normalized confusion matrix
# plt.figure()
# plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#                       title='Normalized confusion matrix')

# plt.show()



if __name__ == "__main__":
	setting = edict();
	setting.gpu 			= 1;
	setting.db 				= cfg.db.gtea;

	db = Db(setting.db, cfg.dstDir);
	db.genDb();

	dataPath = '/home/bbu/Workspace/data/GTEA/';

	act_classes = open(os.path.join(dataPath, 'activity_label.txt')).read().split('\n')[:-1];
	mtn_classes = open(os.path.join(dataPath, 'motion_label.txt')).read().split('\n')[:-1];
	obj_classes = open(os.path.join(dataPath, 'object_label.txt')).read().split('\n')[:-1];

	a = open(os.path.join(dataPath, 'activity_label.txt')).read().split('\n')[:-1];

	print len(act_classes), len(mtn_classes), len(obj_classes)


	for testid in [3]:
		filePath = 'data/result/confusion_matrix/object/predict_result_%d.npy' % testid
		# filePath = 'data/result/confusion_matrix/predict_result_3.npy';

		array = np.load(filePath);

		print array.shape
		# obj_y = array[0,:];
		mtn_y = array[0,:];
		# act_y = array[2,:];

		# obj_predict = array[3,:];
		mtn_predict = array[1,:];
		# act_predict = array[5,:];

		# print obj_y.shape
		# print obj_predict.shape
		cnf_matrix = confusion_matrix(mtn_y, mtn_predict);
		np.set_printoptions(precision=2)

		plt.figure()
		plot_confusion_matrix(cnf_matrix, classes=obj_classes, normalize=True, title='Object Recognition Results for GTEA');
		plt.show()
	# print db.dstDir