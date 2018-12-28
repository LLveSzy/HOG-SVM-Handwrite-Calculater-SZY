# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:10:57 2018

@author: DeltaS
"""

import cv2
from sklearn.externals import joblib
from skimage.feature import hog
import numpy as np
import utils


def pre_img_1(im,bush_points):               
    clf = joblib.load("digits_cls_ex.pkl")                 
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)      
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)      
    
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)             
    binary,ctrs, hira = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]   
    left_to_right = []
    
    for rect in rects:
        left_to_right.append([rect[0],rect[1],rect[2],rect[3]])
    
    #找每一笔画的左右最大值最小值
    mxmn = [] #最小值 最大值 下标
    cnt = 0
    for p in bush_points:
        mxmn.append([np.min(np.array(p)[:,0]),np.max(np.array(p)[:,0]),cnt,np.min(np.array(p)[:,1]),np.max(np.array(p)[:,1])])
        cnt = cnt + 1
    
    
    #连通域
    left_to_right = sorted(left_to_right, key=lambda left_to_right : left_to_right[:][0])
#    bush_points = sorted(bush_points,key=lambda bush_points : bush_points[:][-1][0])
#    print (left_to_right)
    pred_res = []
    print(left_to_right)
    for region in left_to_right:
        #找到范围之内的笔画
        ls = []
        for i in mxmn:
            if i[0] >= region[0] and i[1] <= region[0]+region[2]: #考虑在相同x范围之内但是并不连通
                ls.append(i)
                if i[4] < region[1]  or i[3] > region[1] + region[3]:  #如果并不连通
                    del left_to_right[0]
        
        ls = sorted(ls ,key=lambda ls : (ls[:][0] + ls[:][1])/2)

        idx = 0

        while idx < len(ls):
            result = []
            result.append(bush_points[ls[idx][2]])
            idx = idx + 1 
            
            while idx <  len(ls) and ((ls[idx][0] + ls[idx][1]) - (ls[idx-1][0] + ls[idx-1][1]))/2 \
                                    /(max(ls[idx][1],ls[idx-1][1]) - min(ls[idx][0],ls[idx-1][0])) < 0.4: #下一笔仍在范围之内
                result.append(bush_points[ls[idx][2]])
                idx = idx + 1
                
        
                
            pred_res.append(pred(utils.handl_img(utils.proc_array(result)))) 
                     
#        for pl in np.array(ls)[:,2
 #            result.append(bush_points[pl]) #点序列
#        for plst in result:
#            pred_res.append(pred(utils.handl_img(utils.proc_array(plst))))
#        image = im[region[1]:(region[1]+region[3]),region[0]:(region[0]+region[2]),:]    
#        img = utils.handl_img(255-image)  
#        result.append(pred(img))
    return pred_res
#    return imgs

def pred(im):
    roi = cv2.dilate(im, (3, 3))
    clf = joblib.load("digits_cls_ex.pkl")
    roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)  
    nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
    return str(nbr[0])
    
def pre_img(im):                      
    clf = joblib.load("digits_cls_ex.pkl")                 
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)      
    im_gray = cv2.GaussianBlur(im_gray, (5, 5), 0)      
    
    ret, im_th = cv2.threshold(im_gray, 90, 255, cv2.THRESH_BINARY_INV)             
    binary,ctrs, hira = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]    
    
    for rect in rects:
        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3) 
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        
        roi = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi = cv2.dilate(roi, (3, 3))
        
        roi_hog_fd = hog(roi, orientations=9, pixels_per_cell=(14, 14), cells_per_block=(1, 1), visualise=False)  
        nbr = clf.predict(np.array([roi_hog_fd], 'float64'))
        print(nbr)
        pre = int(nbr[0])
        out = ''
        if pre == 10:
            out = '('
        elif pre == 11:
            out = ')'
        elif pre == 12:
            out = '/'
        elif pre == 13:
            out = 'X'
        elif pre == 14:
            out = '+'
        elif pre == 15:
            out = '-'
        else:
            out = str(pre)
        cv2.putText(im, out, (rect[0], rect[1]),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 255), 3)
        print(clf.predict_proba(np.array([roi_hog_fd], 'float64')))
    cv2.imshow('image', im)

