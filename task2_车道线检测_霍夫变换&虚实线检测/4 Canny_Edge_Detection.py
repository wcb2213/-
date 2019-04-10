#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2018/12/12


import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import cv2

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 75  # Canny edge detection low threshold
canny_hthreshold = 125  # Canny edge detection high threshold
img = mplimg.imread('./cut_image/5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
plt.imshow(edges)
plt.show()