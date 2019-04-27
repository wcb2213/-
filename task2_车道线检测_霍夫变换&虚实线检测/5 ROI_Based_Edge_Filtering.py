#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2018/12/12


import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np
import cv2

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold
def roi_mask(img, vertices):
  mask = np.zeros_like(img)
  mask_color = 255
  cv2.fillPoly(mask, vertices, mask_color)
  masked_img = cv2.bitwise_and(img, mask)
  return masked_img
img = mplimg.imread('./cut_imgs/5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
roi_vtx = np.array([[(100, img.shape[0]-150), (460, 375),
                     (520, 375), (img.shape[1]-100, img.shape[0]-150)]])
print(roi_vtx)
# roi_vtx = np.array([[(0, img.shape[0]), (460, 325),
#                      (520, 325), (img.shape[1], img.shape[0])]])
roi_edges = roi_mask(edges, roi_vtx)
plt.imshow(roi_edges)
plt.show()