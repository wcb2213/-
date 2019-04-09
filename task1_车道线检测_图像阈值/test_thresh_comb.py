#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2018/12/9

import numpy as np
import cv2
from thresholds import mag_thresh, dir_thresh, color_thresh


img = cv2.imread('./cut_image/3.jpg')

# Threshold gradient
grad_binary = np.zeros_like(img[:, :, 0])
# mag_binary = mag_thresh(img, sobel_kernel=9, thresh=(50, 255))
mag_binary = mag_thresh(img, sobel_kernel=3, thresh=(20, 255))
# dir_binary = dir_thresh(img, sobel_kernel=15, thresh=(0.7, 1.3))
dir_binary = dir_thresh(img, sobel_kernel=15, thresh=(0.55, 1.3))
grad_binary[((mag_binary == 1) & (dir_binary == 1))] = 1

# Threshold color
####    Combine the two channels with |
# color_binary = color_thresh(img, r_thresh=(220, 255), s_thresh=(150, 255))
color_binary = color_thresh(img, r_thresh=(125, 190), s_thresh=(14, 18))
# Combine gradient and color thresholds
combo_binary = np.zeros_like(img[:, :, 0])
combo_binary[(grad_binary == 1) | (color_binary == 1)] = 255


cv2.imshow("combo_binary", combo_binary)

cv2.waitKey(0)
cv2.destroyAllWindows()