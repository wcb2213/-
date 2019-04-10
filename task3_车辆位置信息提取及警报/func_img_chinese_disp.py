#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2018/12/17


from PIL import Image,ImageDraw,ImageFont
import cv2
import numpy as np
import matplotlib.pyplot as plt

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

