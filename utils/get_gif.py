#!/usr/local/bin
# -*- coding: utf-8 -*-
# @Author: Jingrou Wu
# @Date:   2018-12-24 18:21:38
# @Last Modified by:   Jingrou Wu
# @Last Modified time: 2018-12-24 18:45:42

from PIL import Image
from images2gif import writeGif
import imageio
import numpy as np


path = "pic/fig_%d.png"
outfilename = "rule30.gif"  # 转化的GIF图片名称
filenames = []         # 存储所需要读取的图片名称
for i in range(26):   # 读取100张图片
    filename = path % (i + 1)    # path是图片所在文件，最后filename的名字必须是存在的图片
    filenames.append(filename)              # 将使用的读取图片汇总
images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('rule30.gif', images, duration=1)
