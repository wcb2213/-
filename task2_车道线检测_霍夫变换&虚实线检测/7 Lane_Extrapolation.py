#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2018/12/12

import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from img_chinese_disp import img_chinese_display

blur_ksize = 5  # Gaussian blur kernel size
canny_lthreshold = 50  # Canny edge detection low threshold
canny_hthreshold = 150  # Canny edge detection high threshold
# Hough transform parameters
rho = 1
theta = np.pi / 180
threshold = 15
min_line_length = 40
max_line_gap = 20

# 添加
color2=[0, 255, 0]
color1=[255, 0, 0]

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
    # print(lines)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    # draw_lines(line_img, lines)
    # draw_lanes(line_img, lines)
    return draw_lanes(line_img, lines)


def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):
    left_lines, right_lines = [], []
    for line in lines:
        for x1, y1, x2, y2 in line:
            k = (y2 - y1) / (x2 - x1)
            if k < 0:
                left_lines.append(line)
            else:
                right_lines.append(line)

    if (len(left_lines) <= 0 or len(right_lines) <= 0):
        return img
    clean_lines(left_lines, 0.1)
    clean_lines(right_lines, 0.1)
    # print(left_lines)
    # [array([[242, 570, 536, 380]], dtype=int32), array([[227, 570, 525, 376]], dtype=int32),
    #  array([[244, 570, 535, 381]], dtype=int32), array([[228, 570, 272, 542]], dtype=int32),
    #  array([[474, 411, 525, 377]], dtype=int32)]
    # print(right_lines)
    # [array([[616, 403, 846, 570]], dtype=int32), array([[684, 463, 831, 570]], dtype=int32),
    #  array([[753, 516, 824, 570]], dtype=int32), array([[595, 397, 695, 470]], dtype=int32),
    #  array([[644, 423, 845, 569]], dtype=int32), array([[761, 523, 818, 566]], dtype=int32),
    #  array([[652, 439, 727, 494]], dtype=int32), array([[630, 414, 681, 451]], dtype=int32)]

    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    # print(left_points)
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    # print(left_points)
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    # print(right_points)
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]
    # print(right_points)
    left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
    print(left_vtx)
    # [(594, 325), (-40, 720)]
    right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])
    print(right_vtx)
    # [(514, 325), (989, 720)]
    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)
    # image = pianliyujing(img, left_vtx, right_vtx)
    return xushixian(img, left_lines, right_lines, left_vtx, right_vtx)

# def pianliyujing(img, left_vtx, right_vtx):
#     center = (left_vtx[0][0]+right_vtx[0][0])/2
#     vehicle_center = 500
#     if vehicle_center - center >= 0:
#         img_chinese_display(img, '偏右：' + str(vehicle_center - center) + ' 像素', color=(0, 0, 255), position=(50, 50))
#     else:
#         img_chinese_display(img, '偏左：' + str(center - vehicle_center) + ' 像素', color=(0, 0, 255), position=(50, 50))

def xushixian(img, left_lines, right_lines, left_vtx, right_vtx):
    # 添加
    # left_lines[0][0][0]= x1, left_lines[0][0][1]= y1 霍夫变换出来的 x1必然小于x2
    L1 = len(left_lines)
    k1 = 0
    for i in range(0, L1):
        if np.sqrt(np.square(left_lines[i][0][0] - left_lines[i][0][2]) + np.square(
                left_lines[i][0][1] - left_lines[i][0][3])) > 208:
            k1 = k1 + 1
        else:
            k1 = k1
    if k1 >= 1:
        img = img_chinese_display(img, '左车道：实线', color=(0, 0, 255), position=(50, 100))
    else:
        img = img_chinese_display(img, '左车道：虚线', color=(0, 0, 255), position=(50, 100))
    print(k1)


    L2 = len(right_lines)
    k2 = 0
    for i in range(0, L2):
        if np.sqrt(np.square(right_lines[i][0][0] - right_lines[i][0][2]) + np.square(
                right_lines[i][0][1] - right_lines[i][0][3])) > 208:
            k2 = k2 + 1
        else:
            k2 = k2
    if k2 >= 1:
        img = img_chinese_display(img, '右车道：实线', color=(0, 0, 255), position=(50, 150))
    else:
        img = img_chinese_display(img, '右车道：虚线', color=(0, 0, 255), position=(50, 150))
    print(k2)

    center = (left_vtx[0][0]+right_vtx[0][0])/2
    vehicle_center = 550
    if vehicle_center - center >= 0:
        img = img_chinese_display(img, '偏右：' + str(vehicle_center - center) + ' 像素', color=(0, 0, 255), position=(50, 50))
    else:
        img = img_chinese_display(img, '偏左：' + str(center - vehicle_center) + ' 像素', color=(0, 0, 255), position=(50, 50))


    a = vehicle_center - left_vtx[0][0]
    b = vehicle_center - right_vtx[0][0]
    if (abs(a) < 20) | (abs(b) < 20):
        img = img_chinese_display(img, '越道中', color=(0, 0, 255),
                                  position=(50, 200))
    return img


def clean_lines(lines, threshold):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop(idx)
        else:
            break


def calc_lane_vertices(point_list, ymin, ymax):
    x = [p[0] for p in point_list]
    y = [p[1] for p in point_list]
    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)
    xmin = int(fit_fn(ymin))
    xmax = int(fit_fn(ymax))
    return [(xmin, ymin), (xmax, ymax)]

img = mplimg.imread('./cut_image/12.jpg')
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
