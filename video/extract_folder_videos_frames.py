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

videoFolder = "/media/chtseng/WIN7"
framesSavePath = "/media/sf_VMShare/cars/"
resizeWidth = 0
rotate = 0
interval = 10 #frames
#----------------------------------------
if not os.path.exists(framesSavePath):
    os.makedirs(framesSavePath)

camera = None
for vfile in os.listdir(videoFolder):
    print(vfile)
    filename, file_extension = os.path.splitext(vfile)
    file_extension = file_extension.lower()

    if(file_extension == ".mp4" or file_extension==".avi" or file_extension==".mpeg" or file_extension==".mov"):
        if(camera is not None):
            camera.release()

        camera = cv2.VideoCapture(os.path.join(videoFolder, vfile))
        extract_folder = os.path.join(framesSavePath, filename)
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        grabbed = True
        i = 0
        frameid = 0
        while(grabbed is True):
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

                    cv2.imwrite(os.path.join(extract_folder,filename), img)
                    print("{} saved.".format(filename))

                    i += 1

                frameid += 1
