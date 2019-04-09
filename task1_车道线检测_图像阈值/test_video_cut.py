#!/usr/bin/env/ python
# -*- coding:utf-8 -*-
# Created by: Vanish
# Created on: 2018/12/9

from moviepy.editor import *

# 截取一段5秒的视频
clip1 = VideoFileClip('./video_1.mp4').subclip(100,105)
clip1.write_videofile('video_1_cut.mp4', audio=False)