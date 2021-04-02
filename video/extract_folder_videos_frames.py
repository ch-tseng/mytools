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
from tqdm import tqdm

videoFolder = r"D:\Projects\ModelSale_Humancrowded_swimming\DataSource\videos"
framesSavePath = r"D:\Projects\ModelSale_Humancrowded_swimming\DataSource\extrated_frames"
resizeWidth = 960
rotate = 0
interval = 30 #frames
#----------------------------------------

videoFolder = videoFolder.replace('\\', '/')
framesSavePath = framesSavePath.replace('\\', '/')

if not os.path.exists(framesSavePath):
    os.makedirs(framesSavePath)

camera = None
for id, vfile in tqdm(enumerate(os.listdir(videoFolder))):
    #print(vfile)
    filename, file_extension = os.path.splitext(vfile)
    file_extension = file_extension.lower()

    if(file_extension.lower() in [".mp4",".avi",".mpeg",".mov"]):
        if(camera is not None):
            camera.release()

        now = datetime.datetime.now()
        append_prefix_filename = "{}{}{}{}{}{}".format(now.year,now.month,now.day,now.hour,now.minute,now.second)

        camera = cv2.VideoCapture(os.path.join(videoFolder, vfile))
        length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))
        extract_folder = os.path.join(framesSavePath, filename)
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        i = 0
        #while(grabbed is True):
        for frameid in tqdm(range(0,length)):
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
                    filename = append_prefix_filename + "_" + str(i) + ".jpg"

                    if(resizeWidth>0 and img.shape[1]>resizeWidth):
                        img = imutils.resize(img, width=resizeWidth)

                    if not os.path.exists(extract_folder):
                        #print("make folder:", extract_folder)
                        os.makedirs(extract_folder)

                    cv2.imwrite(os.path.join(extract_folder,filename), img)
                    #print("{} saved.".format(os.path.join(extract_folder,filename)))

                    i += 1

            (grabbed, img) = camera.read()
