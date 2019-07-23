#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2018/12/12

import matplotlib.image as mplimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image,ImageDraw,ImageFont
from func_img_chinese_disp import img_chinese_display


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
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    left_points = left_points + [(x2, y2) for line in left_lines for x1, y1, x2, y2 in line]
    right_points = [(x1, y1) for line in right_lines for x1, y1, x2, y2 in line]
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]
    left_vtx = calc_lane_vertices(left_points, 325, img.shape[0])
    right_vtx = calc_lane_vertices(right_points, 325, img.shape[0])
    cv2.line(img, left_vtx[0], left_vtx[1], color, thickness)
    cv2.line(img, right_vtx[0], right_vtx[1], color, thickness)
    return xushixian(img, left_lines, right_lines, left_vtx, right_vtx)

#### 添加
def xushixian(img, left_lines, right_lines, left_vtx, right_vtx):
    L1 = len(left_lines)
    k1 = 0
    for i in range(0, L1):
        # left_lines[0][0][0]= x1, left_lines[0][0][1]= y1 霍夫变换出来的 x1必然小于x2
        if np.sqrt(np.square(left_lines[i][0][0] - left_lines[i][0][2]) + np.square(
                left_lines[i][0][1] - left_lines[i][0][3])) > 208:
            k1 = k1 + 1
        else:
            k1 = k1
    if k1 >= 1:
        img = img_chinese_display(img, '左车道：实线', color=(0, 0, 255), position=(50, 100))
    else:
        img = img_chinese_display(img, '左车道：虚线', color=(0, 0, 255), position=(50, 100))
    # print(k1)

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
    # print(k2)

    center = (left_vtx[0][0]+right_vtx[0][0])/2
    vehicle_center = 550
    if vehicle_center - center >= 0:
        img = img_chinese_display(img, '偏右：' + str(vehicle_center - center) + ' 像素', color=(0, 0, 255), position=(50, 50))
    else:
        img = img_chinese_display(img, '偏左：' + str(center - vehicle_center) + ' 像素', color=(0, 0, 255), position=(50, 50))
    return img
    return img


def clean_lines(lines, threshold):
    slope = [(y2 - y1) / (x2 - x1) for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0:
        mean = np.mean(slope)
        diff = [abs(s - mean) for s in slope]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slope.pop(idx)
            lines.pop
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

def img_chinese_display(img, str, color = (255, 0, 0), position = (0, 0)):
    # img = cv2.imread('./out_image/1.jpg')#如想读取中文名称的图片文件可用cv2.imdecode()
    pil_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)#cv2和PIL中颜色的hex码的储存顺序不同，需转RGB模式
    pilimg = Image.fromarray(pil_img)#Image.fromarray()将数组类型转成图片格式，与np.array()相反
    draw = ImageDraw.Draw(pilimg)#PIL图片上打印汉字
    font = ImageFont.truetype("C:\Windows\Fonts\simhei.ttf",50,encoding="utf-8")#参数1：字体文件路径，参数2：字体大小；Windows系统“simhei.ttf”默认存储在路径：C:\Windows\Fonts中
    # position = (0, 0)
    # color = (255, 0, 0)
    draw.text(position,str,color,font=font)
    cv2img = cv2.cvtColor(np.array(pilimg),cv2.COLOR_RGB2BGR)#将图片转成cv2.imshow()可以显示的数组格式
    return cv2img


img = mplimg.imread('./cut_image/5.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
blur_gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0, 0)
edges = cv2.Canny(blur_gray, canny_lthreshold, canny_hthreshold)
roi_vtx = np.array([[(100, img.shape[0]-150), (460, 375),
                     (520, 375), (img.shape[1]-100, img.shape[0]-150)]])
roi_edges = roi_mask(edges, roi_vtx)
line_img = hough_lines(roi_edges, rho, theta, threshold,
                       min_line_length, max_line_gap)
res_img = cv2.addWeighted(img, 0.8, line_img, 1, 0)
# res_img = cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR)

plt.imshow(res_img)
plt.show()

# cv2.imshow("s_thresh", res_img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
