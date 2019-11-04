#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import glob, os
from imutils import paths
from scipy import io
import numpy as np
import imutils
import cv2
import sys
import time
import datetime

videoFile = "/Users/chtseng/works/IMG_3492.mov"
framesSavePath = "/Users/chtseng/Downloads/vehicles"
append_prefix_filename = "a"
resizeWidth = 0
rotate = 0
interval = 72 #frames
#----------------------------------------
if not os.path.exists(framesSavePath):
    os.makedirs(framesSavePath)

camera = cv2.VideoCapture(videoFile)

i = 0
frameid = 0
while(camera.isOpened()):
    (grabbed, img) = camera.read()
    if(rotate>0):
        img = imutils.rotate_bound(img, rotate)


    if(grabbed is True):
        cv2.imshow("Frame", imutils.resize(img, width=300))
        k = cv2.waitKey(1)
        if(k==113):
            break

        if(frameid % interval == 0):
            filename = str(i).zfill(8)
            filename = append_prefix_filename + "_" + filename + ".jpg"

            if(resizeWidth>0):
                img = imutils.resize(img, width=resizeWidth)

            cv2.imwrite(os.path.join(framesSavePath,filename), img)
            print("{} saved.".format(filename))

            i += 1

        frameid += 1
