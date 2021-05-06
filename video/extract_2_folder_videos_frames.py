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
import pathlib

videoFolder = r"/media/chtseng/WORKS/Media_Sources/Videos/科技執法/違規檢舉影片"
framesSavePath = r"/WORKS/working/law"
resizeWidth = 960
rotate = 0
interval = 10 #frames
#----------------------------------------

videoFolder = videoFolder.replace('\\', '/')
framesSavePath = framesSavePath.replace('\\', '/')

if not os.path.exists(framesSavePath):
    os.makedirs(framesSavePath)

camera = None

for subf in tqdm(os.listdir(videoFolder)):
    subfolder = os.path.join(videoFolder, subf)
    if not os.path.isdir(subfolder):
        continue

    for id, vfile in enumerate(os.listdir(subfolder)):
        print(vfile)
        filename, file_extension = os.path.splitext(vfile)
        file_extension = file_extension.lower()

        if(file_extension.lower() in [".mp4",".avi",".mpeg",".mov"]):
            if(camera is not None):
                camera.release()

            now = datetime.datetime.now()
            append_prefix_filename = "{}{}{}{}{}{}".format(now.year,now.month,now.day,now.hour,now.minute,now.second)

            camera = cv2.VideoCapture(os.path.join(subfolder, vfile))
            length = int(camera.get(cv2.CAP_PROP_FRAME_COUNT))

            filename = filename.replace(' ', '')
            filename = filename.replace("'", '')
            filename = filename.replace("\\", '')
            filename = filename.replace(".", '')
            filename = filename.replace("\"", '')
            filename = filename.replace("“",'')
            filename = filename.replace("”",'')
            extract_folder = os.path.join(framesSavePath, subf, filename)
            if not os.path.exists(extract_folder):
                os.makedirs(extract_folder)

            i, frameid = 0, 0
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
                        filename = append_prefix_filename + "_" + str(i) + ".jpg"

                        if(resizeWidth>0 and img.shape[1]>resizeWidth):
                            img = imutils.resize(img, width=resizeWidth)

                        if not os.path.exists(extract_folder):
                            print("make folder:", extract_folder)
                            pathlib.Path(extract_folder).mkdir(parents=True, exist_ok=True) 
                            #os.makedirs(extract_folder)

                        cv2.imwrite(os.path.join(extract_folder,filename), img)
                        print('write to', os.path.join(extract_folder,filename))

                        i += 1
                frameid += 1
                (grabbed, img) = camera.read()
