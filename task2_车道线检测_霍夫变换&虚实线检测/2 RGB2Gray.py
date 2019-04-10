#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2018/12/12


import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import cv2

img = mplimg.imread('./cut_image/1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
plt.imshow(gray)
plt.show()