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
# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20


def roi_mask(img, vertices):
    mask = np.zeros_like(img)
    mask_color = 255
    cv2.fillPoly(mask, vertices, mask_color)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def hough_lines(img, rho, theta, threshold,
                min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                            minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    print(lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


img = mplimg.imread('./cut_image/5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
roi_vtx = np.array([[(100, img.shape[0]-150), (460, 375),
                     (520, 375), (img.shape[1]-100, img.shape[0]-150)]])
roi_edges = roi_mask(edges, roi_vtx)
line_img = hough_lines(roi_edges, rho, theta, threshold,
                       min_line_length, max_line_gap)
plt.imshow(line_img)
plt.show()