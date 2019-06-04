#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom

#-------------------------------------------

mtcnn_label_file = "/media/sf_VMShare/mtcnn_faces.txt"
dataset_images = "/media/sf_VMShare/Eden/20190522/image/"
dataset_labels = "/media/sf_VMShare/Eden/20190522/50cm_500pics_result/"

#-------------------------------------------

def chkEnv():
    if(not os.path.exists(dataset_images)):
        print("There is no such folder {}".format(dataset_images))
        quit()

    if(not os.path.exists(dataset_labels)):
        print("There is no such folder {}".format(dataset_labels))
        quit()

def getLabels(imgFile, xmlFile):
    labelXML = minidom.parse(xmlFile)
    labelName = []
    labelXmin = []
    labelYmin = []
    labelXmax = []
    labelYmax = []
    totalW = 0
    totalH = 0
    countLabels = 0

    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        labelName.append(str(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmin")
    for elem in tmpArrays:
        labelXmin.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymin")
    for elem in tmpArrays:
        labelYmin.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmax")
    for elem in tmpArrays:
        labelXmax.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymax")
    for elem in tmpArrays:
        labelYmax.append(int(elem.firstChild.data))

    return labelName, labelXmin, labelYmin, labelXmax, labelYmax

def write_lale_images(label, img, saveto, filename):
    writePath = os.path.join(extract_to, label)
    print("WRITE:", writePath)

    if not os.path.exists(writePath):
        os.makedirs(writePath)

    cv2.imwrite(os.path.join(writePath, filename), img)

#--------------------------------------------

chkEnv()

i = 0

file_mtcnn = open(mtcnn_label_file, 'w')

for file in os.listdir(dataset_images):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        print("Processing: ", os.path.join(dataset_images,file))

        if not os.path.exists(dataset_labels+filename+".xml"):
            print("Cannot find the file {} for the image.".format(os.path.join(dataset_labels,filename+".xml")))

        else:
            image_path = os.path.join(dataset_images, file)
            xml_path = os.path.join(dataset_labels, filename+".xml")
            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(image_path, xml_path)

            #orgImage = cv2.imread(image_path)
            #image = orgImage.copy()
            line_mtcnn = image_path

            if('face' in labelName):
                i = labelName.index('face')
                p1 = labelXmin[i]   #left-top point(x) of the face
                p2 = labelXmax[i]   #right-top point(x) of the face
                p3 = labelYmin[i]   #right-top point(y) of the face
                p4 = labelYmax[i]   #left-bottom point(y) of the face
            else:
                print("ERROR: {} has no label for face".format(image_path))
                continue

            if('reye_010' in labelName):
                i = labelName.index('reye_010')
                p5 = int((labelXmin[i] + labelXmax[i])/2)   #center point(x) of the right eye
                p6 = int((labelYmin[i] + labelYmax[i])/2)   #center point(y) of the right eye
            else:
                print("ERROR: {} has no label for right eye".format(image_path))
                continue

            if('leye_010' in labelName):
                i = labelName.index('leye_010')
                p7 = int((labelXmin[i] + labelXmax[i])/2)   #center point(x) of the left eye
                p8 = int((labelYmin[i] + labelYmax[i])/2)   #center point(y) of the left eye
            else:
                print("ERROR: {} has no label for left eye".format(image_path))
                continue

            if('nose' in labelName):
                i = labelName.index('nose')
                p9 = int((labelXmin[i] + labelXmax[i])/2)   #point(x) of the nose top
                p10 = int((labelYmin[i] + labelYmax[i])/2)   #point(y) of the nose top
            else:
                print("ERROR: {} has no label for nose".format(image_path))
                continue

            if('rmouth' in labelName):
                i = labelName.index('rmouth')
                p11 = int((labelXmin[i] + labelXmax[i])/2)   #point(x) of the right mouth
                p12 = int((labelYmin[i] + labelYmax[i])/2)   #point(y) of the right mouth
            else:
                print("ERROR: {} has no label for right mouth".format(image_path))
                continue

            if('lmouth' in labelName):
                i = labelName.index('lmouth')
                p13 = int((labelXmin[i] + labelXmax[i])/2)   #point(x) of the left mouth
                p14 = int((labelYmin[i] + labelYmax[i])/2)   #point(y) of the left mouth
            else:
                print("ERROR: {} has no label for left mouth".format(image_path))
                continue

            file_mtcnn.write("{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n"
                    .format(image_path, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14) )


file_mtcnn.close()
