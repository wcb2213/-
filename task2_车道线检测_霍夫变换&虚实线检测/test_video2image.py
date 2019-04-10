#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2018/12/9

import cv2

vc = cv2.VideoCapture('my_test_video.mp4')
c = 1

if vc.isOpened():
    rval,frame = vc.read()
else:
    rval = False

timeF = 15

while rval:
    rval, frame = vc.read()
    if(c%timeF == 0):
        cv2.imwrite('cut_image\\'+str(int(c/15))+'.jpg',frame)
    c = c+1
    cv2.waitKey(1)
vc.release()