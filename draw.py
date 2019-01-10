# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 22:54:59 2018

@author: DeltaS
"""

import cv2
import numpy as np
import predict
import utils
   
    
def mouse_event(event, x, y, flags, param): 
    global start, drawing,x0,y0,img,rects,bush_poly,bush_points,points
    
    if event == cv2.EVENT_LBUTTONDOWN:
        x0 = x
        y0 = y
        drawing = True
        points.append([x,y])
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            points.append([x,y])
            cv2.line(img,(x0,y0),(x,y),(0,0,0),3)
            x0 = x
            y0 = y 
    elif event == cv2.EVENT_LBUTTONUP:
        points.append([x,y])
        bush_points.append(points)
#        poly_img = utils.proc_array(points)
#        d = utils.handl_img(poly_img)
#        predict.pred(d)
#        bush_poly.append(255-d)
#        cv2.imshow("handle",handl_img(poly_img))
#        cv2.waitKey(0)
#        cv2.destroyWindow("handle")
#        cv2.imshow("poly",poly_img)
#        cv2.waitKey(0)
#        cv2.destroyWindow("poly")
        points = []
        drawing = False       
    
    
if __name__ == "__main__":
    drawing = False  
    x0 = 0
    y0 = 0
    cv2.namedWindow('image')
    img = np.zeros((512, 700, 3), np.uint8) + 255
    
    rects = []
    bush_poly = []
    bush_points = []
    points = []
#    print('thread %s is running...' % threading.current_thread().name)
#    t = threading.Thread(target=main(), name='sendThread')
#    t.start()
#    t.join()  
    cv2.setMouseCallback('image', mouse_event)
    while(True):  
        cv2.imshow('image', img)
        if cv2.waitKey(1) == 27:
            break
        
        elif cv2.waitKey(1)& 0xFF == ord('q'):
#            predict.pre_img(img)
            result = predict.pre_img_1(img,bush_points)
            print(result)
            rs = ''
            for pre in result:
                pre = int(pre)
                if  pre == 10:
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
                elif pre == 16:
                    out = '.'
                else:
                    out = str(pre)
                rs = rs + out
            print(rs)
            
            #计算结果
#            rs = rs + "=" + str(utils.postfix_calculate(utils.middle2behind(rs)))
            
            tp = utils.middle2behind(rs)
            res = str(utils.postfix_calculate(tp)) if (tp != False) else '?' 
            font_size = 2 if(30 + 40*len(rs + "=" + res) < 700) else 1
            cv2.putText(img,rs + "=" + res, (30, 400),cv2.FONT_HERSHEY_DUPLEX, font_size, (0, 0, 255), 2)
            
        elif cv2.waitKey(1)& 0xFF == ord('w'):
            img = np.zeros((512, 700, 3), np.uint8) + 255
            rects = []
            bush_poly = []
            bush_points = []
            points = []
    cv2.destroyAllWindows()     
#def main():
#    hm = pyHook.HookManager() 
#    hm.KeyDown = onKeyboardEvent
#    hm.HookKeyboard()
#    
##    hm.MouseLeftDown = onMouse_leftdown
##    hm.MouseLeftUp = onMouse_leftup
##    hm.MouseMove = onMouse_move
##    hm.HookMouse() 
##     进入循环，如不手动关闭，程序将一直处于监听状态
#    pythoncom.PumpMessages()
#    while(True):        
#        image_show(img)
#        cv2.waitKey(0)

    
    



            

    