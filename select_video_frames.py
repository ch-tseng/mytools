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

preChar = "a"  #any character, this will add to the jpg's filename as first char.
videoFile = "0"  # video's path, or 0 is for web cam
webCamSize = (1920, 1080)  #for web cam only
framesSavePath = "bread_0/"
resizeWidth = 0
rotate = 90
#----------------------------------------
if not os.path.exists(framesSavePath):
    os.makedirs(framesSavePath)

if(videoFile.isdigit() is True):
    camera = cv2.VideoCapture(int(videoFile))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, webCamSize[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, webCamSize[1])
else:
    camera = cv2.VideoCapture(videoFile)

i = 0
while(camera.isOpened()):
    (grabbed, img) = camera.read()
    if(rotate>0):
        img = imutils.rotate_bound(img, rotate)


    if(grabbed is True):
        cv2.imshow("Frame", imutils.resize(img, width=300))
        k = cv2.waitKey(1)
        if(k == 99):
            filename = preChar + "_" + str(i).zfill(8)
            filename = filename + ".jpg"
            if(resizeWidth>0):
                img = imutils.resize(img, width=resizeWidth)

            cv2.imwrite(framesSavePath + filename, img)
            print("{} saved.".format(filename))
            i += 1

