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

videoFolder = "/Volumes/AIDATA1/fruits/"
framesSavePath = "/Volumes/AIDATA1/fakeFruits/"
append_prefix_filename = "all"
resizeWidth = 0
rotate = 270
interval = 45 #frames
#----------------------------------------
if not os.path.exists(framesSavePath):
    os.makedirs(framesSavePath)

camera = None
for id, vfile in enumerate(os.listdir(videoFolder)):
    print(vfile)
    filename, file_extension = os.path.splitext(vfile)
    file_extension = file_extension.lower()

    if(file_extension.lower() in [".mp4",".avi",".mpeg",".mov"]):
        if(camera is not None):
            camera.release()

        camera = cv2.VideoCapture(os.path.join(videoFolder, vfile))
        extract_folder = os.path.join(framesSavePath, filename)
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        i = 0
        frameid = 0
        (grabbed, img) = camera.read()
        while(grabbed is True):
            if(rotate>0):
                img = imutils.rotate_bound(img, rotate)


            if(grabbed is True):
                cv2.imshow("Frame", imutils.resize(img, width=300))
                k = cv2.waitKey(1)
                if(k==113):
                    break

                if(frameid % interval == 0):
                    filename = str(i).zfill(8)
                    filename = append_prefix_filename + "_" + str(id) + "_" + filename + ".jpg"

                    if(resizeWidth>0 and img.shape[1]>resizeWidth):
                        img = imutils.resize(img, width=resizeWidth)

                    if not os.path.exists(extract_folder):
                        print("make folder:", extract_folder)
                        os.makedirs(extract_folder)

                    cv2.imwrite(os.path.join(extract_folder,filename), img)
                    print("{} saved.".format(os.path.join(extract_folder,filename)))

                    i += 1

                frameid += 1
            (grabbed, img) = camera.read()
