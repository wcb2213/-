#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2019/1/6


import cv2 as cv
import matplotlib.image as mplimg
import matplotlib.pyplot as plt

# src=cv.imread('test_images/frame007.jpg')
# cv.namedWindow('input_image', cv.WINDOW_AUTOSIZE)
# cv.imshow('input_image', src)
# cv.waitKey(0)
# cv.destroyAllWindows()



img = mplimg.imread('cut_image/test.jpg')
# gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(img)
plt.show()