# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 15:28:55 2018

@author: DeltaS
"""

from sklearn.externals import joblib
from sklearn import datasets
from skimage.feature import hog
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
import numpy as np
import scipy.io as sio
import os
import struct
import cv2

rootdir = "./cfs/"

load_data = sio.loadmat('./mldata/mnist-extend.mat')
features = np.array(load_data['data'], 'int16') 
labels = np.array(load_data['target'], 'int').T

def extend_mnist(syb,lab):
    global dt,lb
    list = os.listdir(syb)      
    for cnt in range(0,len(list)):
        path = os.path.join(syb,list[cnt])
        if os.path.isfile(path):
            img = (255 - cv2.imread(syb + str(cnt+1) + ".jpg",0)).reshape((1,784))
            dt = np.row_stack((dt,img))
            a = np.array([lab]).astype(np.float64)
            lb = np.r_[lb,a] 
            
def proc(rootdir):
    
    list = os.listdir(rootdir) 
    for cnt in range(0,len(list)):       
        path = os.path.join(rootdir,list[cnt]) + "/"
        print(path) 
        extend_mnist(path,10+cnt)

# --- 扩充mnist -- #
#dataset = datasets.fetch_mldata("MNIST Original",data_home=r'.')
#features = np.array(dataset.data, 'int16') 
#labels = np.array(dataset.target, 'int')
#dt = dataset.data
#lb = dataset.target
#proc(rootdir)
#sio.savemat("./mldata/mnist-extend.mat",{'data':dt,'target':lb})


#def load_mnist(path):            #读取数据函数
#    #Load MNIST data from path
#    labels_path = os.path.join(path, 'train-labels.idx1-ubyte')
#    images_path = os.path.join(path, 'train-images.idx3-ubyte')
#
#    with open(labels_path, 'rb') as lbpath:
#        magic, n = struct.unpack('>II',lbpath.read(8))
#        labels = np.fromfile(lbpath, dtype=np.uint8)
#
#    with open(images_path, 'rb') as imgpath:
#        magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
#        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
#    return images, labels
#
#features,labels = load_mnist("./")
#print('Rows: %d, columns: %d' % (features.shape[0], labels.shape[0]))

#def proc_img(img):
    


#---training------#
list_hog_fd = [] 
for feature in features:
    fd = hog(feature.reshape((28, 28)),     # hog 特征
             orientations=9, 
             pixels_per_cell=(14, 14), 
             cells_per_block=(1, 1), 
             visualise=False)
    list_hog_fd.append(fd)
hog_features = np.array(list_hog_fd, 'float64')

clf =svm.SVC(C=0.7, cache_size=200, class_weight=None, coef0=0.0, 
    decision_function_shape='ovo', degree=5, gamma='auto_deprecated',
    kernel='rbf', max_iter=-1, probability=True, random_state=None,
    shrinking=True, tol=0.04, verbose=False)                          

clf.fit(hog_features, labels)                    
joblib.dump(clf, "digits_cls_ex.pkl", compress=3)  

# 压缩：0到9的整数可选
# 压缩级别：0没有压缩。越高意味着更多的压缩，而且读取和写入越慢。使用3的值通常是一个很好的折衷。
