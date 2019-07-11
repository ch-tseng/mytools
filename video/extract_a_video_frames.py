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

videoFile = "/media/sf_VMShare/P1180596.MP4"
framesSavePath = "/media/sf_VMShare/piano/"
resizeWidth = 0
rotate = 0
interval = 6 #frames
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
            filename = filename + ".jpg"

            if(resizeWidth>0):
                img = imutils.resize(img, width=resizeWidth)

            cv2.imwrite(os.path.join(framesSavePath,filename), img)
            print("{} saved.".format(filename))

            i += 1

        frameid += 1
