#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2018/12/12


import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import cv2

blur_ksize = 5  # Gaussian blur kernel size
img = mplimg.imread('./cut_image/5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
plt.imshow(blur_gray)
plt.show()