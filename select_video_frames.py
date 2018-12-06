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

videoFile = "videos/IMG_0547.MOV"
framesSavePath = "frames/"
resizeWidth = 0
rotate = 90
#----------------------------------------
if not os.path.exists(framesSavePath):
    os.makedirs(framesSavePath)

camera = cv2.VideoCapture(videoFile)

i = 0
while(camera.isOpened()):
    (grabbed, img) = camera.read()
    if(rotate>0):
        img = imutils.rotate_bound(img, rotate)


    if(grabbed is True):
        cv2.imshow("Frame", imutils.resize(img, width=300))
        k = cv2.waitKey(0)
        if(k == 99):
            filename = str(i).zfill(8)
            filename = filename + ".jpg"
            if(resizeWidth>0):
                img = imutils.resize(img, width=resizeWidth)

            cv2.imwrite(framesSavePath + filename, img)
            print("{} saved.".format(filename))
            i += 1

