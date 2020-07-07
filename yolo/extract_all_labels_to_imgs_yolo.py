#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom

#-------------------------------------------
#extract_to = "H:/Datasets/Weight_Vegetables/extract"
#imgFolder = "H:/Datasets/Weight_Vegetables/images"
#xmlFolder = "H:/Datasets/Weight_Vegetables/labels"
extract_to = "J:/Diabnext_Ritchie/extract"
imgFolder = "J:/Diabnext_Ritchie/images"
yoloFolder = "J:/Diabnext_Ritchie/labels"
class_path = "J:/Diabnext_Ritchie/labels/classes.txt"

resize_to = None  #(32, 32)

#-------------------------------------------

class_list = {}
f = open(class_path, 'r')
line = f.readline()
id = 0
while line:
    line = line.replace('\n','')
    name = line.strip()
    class_list.update( {id:name })
    id += 1
    line = f.readline()
f.close()

print(class_list)

def chkEnv():
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print("no {} folder, created.".format(extract_to))

    if(not os.path.exists(imgFolder)):
        print("There is no such folder {}".format(imgFolder))
        quit()

    if(not os.path.exists(yoloFolder)):
        print("There is no such folder {}".format(yoloFolder))
        quit()

def getLabels(img, yoloFile):
    f = open(yoloFile, 'r')
    line = f.readline()

    labelName, labelXmin, labelYmin, labelXmax, labelYmax = [], [], [], [], []
    while line:
        line = line.replace('\n','')
        datas = line.split(' ')
        
        if(len(datas)==5):
            width, height = img.shape[1], img.shape[0]
            x = int(width*float(datas[1]))
            y = int(height*float(datas[2]))
            w = int(width*float(datas[3]))
            h = int(height*float(datas[4]))
            class_name = class_list[int(datas[0])]
            #print("class:{} x:{}, y:{}, w:{}, h:{}".format(class_name, x, y, w, h))

            labelName.append(class_name)
            labelXmin.append(x)
            labelYmin.append(y)
            labelXmax.append(x+w)
            labelYmax.append(y+h)

        line = f.readline()

    f.close()

    return labelName, labelXmin, labelYmin, labelXmax, labelYmax

def write_lale_images(label, img, saveto, filename):
    writePath = os.path.join(extract_to,label)
    print("WRITE:", writePath)

    if not os.path.exists(writePath):
        os.makedirs(writePath)

    if(resize_to is not None):
        img = cv2.resize(img, resize_to)


    cv2.imwrite(os.path.join(writePath, filename), img)

#--------------------------------------------

chkEnv()

i = 0

for file in os.listdir(imgFolder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        print("Processing: ", os.path.join(imgFolder, file))

        if not os.path.exists(os.path.join(yoloFolder, filename+".txt")):
            print("Cannot find the file {} for the image.".format(os.path.join(yoloFolder, filename+".txt")))

        else:
            image_path = os.path.join(imgFolder, file)
            yolo_path = os.path.join(yoloFolder, filename+".txt")
            orgImage = cv2.imread(image_path)
            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(orgImage, yolo_path)
            print(orgImage.shape)
            print(labelName, labelXmin, labelYmin, labelXmax, labelYmax)

            image = orgImage.copy()
            for id, label in enumerate(labelName):
                cv2.rectangle(image, (labelXmin[id], labelYmin[id]), (labelXmax[id], labelYmax[id]), (0,255,0), 2)
                label_area = orgImage[labelYmin[id]:labelYmax[id], labelXmin[id]:labelXmax[id]]
                label_img_filename = filename + "_" + str(id) + ".jpg"
                write_lale_images(labelName[id], label_area, extract_to, label_img_filename)

            #cv2.imshow("Image", imutils.resize(image, width=700))
            k = cv2.waitKey(1)

