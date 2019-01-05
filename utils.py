# -*- coding: utf-8 -*-
"""
Created on Tue Dec 25 14:31:32 2018

@author: DeltaS
"""
import cv2
import numpy as np
import operator 

def proc_array(points_lst):
    if len(points_lst) == 1: 
        poly = np.array(points_lst[0])
        x0,y0 = np.min(poly[:,0]),np.min(poly[:,1])
        x1,y1 = np.max(poly[:,0]),np.max(poly[:,1])
        brush = max([x1-x0,y1-y0])//8
        poly[:,0] = poly[:,0] - x0 + brush//2
        poly[:,1] = poly[:,1] - y0 +brush//2
        poly_img = np.zeros((y1-y0 + brush, x1-x0 + brush, 3), np.uint8)
        for i in range(len(poly[:,0])-1):       
            cv2.line(poly_img,(poly[i][0],poly[i][1]),(poly[i+1][0],poly[i+1][1]),(255,255,255),brush if(brush > 0) else 1)
        
    else:
        min_x = min_y = 1000
        max_x = max_y = 0
        print(len(points_lst))
        
        for points in points_lst:
            points = np.array(points)
            min_x,min_y = min([min_x,np.min(points[:,0])]), min([min_y,np.min(points[:,1])])
            max_x,max_y = max([max_x,np.max(points[:,0])]), max([max_y,max(points[:,1])])
            
#            min_x,min_y = min(min_x,min(points[0][:])),min(min_y,min(points[1][:]))
#            max_x,max_y = max(max_x,min(points[0][:])),max(max_y,min(points[1][:]))
        brush = max([max_x-min_x,max_y-min_y])//8
        poly_img = np.zeros((max_y - min_y + brush, max_x - min_x + brush, 3), np.uint8)    
        for points in points_lst:
            for i in range(len(points)-1):       
                cv2.line(poly_img,(points[i][0] - min_x + brush//2,points[i][1] - min_y + brush//2), \
                         (points[i+1][0] - min_x +3,points[i+1][1] - min_y + 3),(255,255,255),brush)
    
    return poly_img  
                
            

def handl_img(img):
    w = int(img.shape[1])
    h = int(img.shape[0])
    if w<h:
        margin_h = np.zeros((h//8, w, 3), np.uint8)
        margin_w = np.zeros((h//8*2+h, (h//8*2+h - w)//2, 3), np.uint8)
        
        img = np.row_stack((margin_h, img))
        img = np.row_stack((img, margin_h))
        img = np.column_stack((margin_w, img))
        img = np.column_stack((img, margin_w))
    else:
        margin_w = np.zeros((h, w//8, 3), np.uint8)
        margin_h = np.zeros(((w//8*2+w - h)//2, w//8*2+w, 3), np.uint8)  
        
#        print(margin_h.shape[0],margin_h.shape[1],margin_w.shape[0],margin_w.shape[1],w,h)
        img = np.column_stack((margin_w, img))
        img = np.column_stack((img, margin_w))
        img = np.row_stack((margin_h, img))
        img = np.row_stack((img, margin_h))
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)      
    im_gray = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(im_gray, (28, 28), interpolation=cv2.INTER_AREA)
#    cv2.imshow("a",img)
#    cv2.waitKey(0)
#    cv2.destroyWindow("a")
    return img

class Stack:
    def __init__(self): 
        self.items = []
 
    def isEmpty(self):
        return self.items == []
 
    def push(self, item): 
        self.items.append(item)
 
    def pop(self):
        return self.items.pop()
 
    def peek(self):
        return self.items[len(self.items)-1]
 
    def size(self): 
        return len(self.items)
 
    
def postfix_calculate(s):
    stack = Stack()
    for x in s:
        if str(x).isdigit():
            stack.push(x)
        elif x == "+":
            a = stack.pop()
            b = stack.pop()
            stack.push(float(a)+float(b))
        elif x == "-":
            a = stack.pop()
            b = stack.pop()
            stack.push(float(b)-float(a))
        elif x == "X":
            a = stack.pop()
            b = stack.pop()
            stack.push(float(a)*float(b))
        elif x == "/":
            a = stack.pop()
            b = stack.pop()
            stack.push(float(b)/float(a))
 
    return stack.peek()

def middle2behind(expression):  
    result = ()            # 结果列表
    stack = []             # 栈
    print(expression)
    item = 0
    while item < len(expression): 
        if expression[item].isnumeric():      # 如果当前字符为数字那么直接放入结果列表
            res = int(expression[item])
            while item+1 < len(expression) and expression[item+1].isnumeric() :
                item = item + 1
                res = res * 10 + int(expression[item])  
#            print(res,item)
#            result.append(res)
            result = result + tuple([res])
        else:                     # 如果当前字符为一切其他操作符
            if len(stack) == 0:   # 如果栈空，直接入栈
                stack.append(expression[item])
            elif expression[item] in 'X/(':   # 如果当前字符为*/（，直接入栈
                stack.append(expression[item])
            elif expression[item] == ')':     # 如果右括号则全部弹出（碰到左括号停止）
                if len(stack):
                    t = tuple(stack.pop())
                else:
                    return False
                while operator.eq(t[0],'(') == False:   
#                    result.append(t)
                    result = result + t
                    if len(stack):
                        t = tuple(stack.pop())
                    else:
                        return False
            # 如果当前字符为加减且栈顶为乘除，则开始弹出
            elif expression[item] in '+-' and stack[len(stack)-1] in 'X/':
                if stack.count('(') == 0:           # 如果有左括号，弹到左括号为止     
                    while stack:
#                        result.append(stack.pop())
                        if len(stack):
                            result = result + tuple(stack.pop())
                        else:
                            return False
                else:                               # 如果没有左括号，弹出所有
                    if len(stack):
                        t = tuple(stack.pop())
                    else:
                        return False
                    while operator.eq(t[0],'(') == False:
#                        result.append(t)
                        result = result + t
                        if len(stack):
                            t = (stack.pop())
                        else:
                            return False  
                    stack.append('(')
                stack.append(expression[item])  # 弹出操作完成后将‘+-’入栈
            else:
                stack.append(expression[item])# 其余情况直接入栈（如当前字符为+，栈顶为+-）
        item = item + 1

    # 表达式遍历完了，但是栈中还有操作符不满足弹出条件，把栈中的东西全部弹出
    while stack:
#        result.append(stack.pop())
        result = result + tuple(stack.pop())
    # 返回字符串
    print(result)
    return result

#middle2behind("(12+3)X5-1")