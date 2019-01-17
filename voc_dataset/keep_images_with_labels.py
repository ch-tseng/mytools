# -*- coding: utf-8 -*-

import cv2
from imutils.face_utils import rect_to_bb
#import dlib
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom

#-------------------------------------------

datasetPath = "/media/sf_datasets/categories/breads_POS/"
imgPath = "images/"
labelPath = "labels/"
removedPath = "negatives/"

def chkEnv():
    if not os.path.exists(datasetPath):
        print("There is no dataset folder in this path:", datasetPath)
        quit()

    if not os.path.exists(datasetPath+imgPath):
        print("There is no image folder in this path:", datasetPath + imgPath)
        quit()

    if not os.path.exists(datasetPath+labelPath):
        print("There is no label folder in this path:", datasetPath + labelPath)
        quit()

    if not os.path.exists(datasetPath+removedPath):
        os.makedirs(datasetPath+removedPath)
        os.makedirs(datasetPath+removedPath+"images")
        os.makedirs(datasetPath+removedPath+"labels")
        print("Create the path:", datasetPath + removedPath)


def getLabels(xmlFile):
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

    print("Label count:", len(labelName))
    if(len(labelName)>0):
        return labelName, labelXmin, labelYmin, labelXmax, labelYmax
    else:
        return None, None, None, None, None

#--------------------------------------------

chkEnv()

i = 0
labelFolder = datasetPath + labelPath
imageFolder = datasetPath + imgPath

for file in os.listdir(labelFolder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".xml"):
        label_path = labelFolder + file
        print("Processing: ", label_path)

        if os.path.exists(datasetPath+imgPath+filename+".jpg") or \
                os.path.exists(datasetPath+imgPath+filename+".png") or \
                os.path.exists(datasetPath+imgPath+filename+".bmp") or \
                os.path.exists(datasetPath+imgPath+filename+".JPG") or \
                os.path.exists(datasetPath+imgPath+filename+".jpeg") or \
                os.path.exists(datasetPath+imgPath+filename+".PNG") or \
                os.path.exists(datasetPath+imgPath+filename+".JPEG") or \
                os.path.exists(datasetPath+imgPath+filename+".BMP"):

            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(label_path)

            if(labelName is None):
                print("Empty labels, remove the xml file:", label_path)
                os.rename(label_path, datasetPath+removedPath+"labels/"+file)

        else:
            print("Cannot find the image, remove the xml:{}".format(label_path))
            os.rename(label_path, datasetPath+removedPath+"labels/"+file)


for file in os.listdir(imageFolder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        print("Processing: ", imageFolder + file)
        image_path = imageFolder + file

        if not os.path.exists(datasetPath+labelPath+filename+".xml"):
            print("Cannot find the file {}, remove this.".format(datasetPath+labelPath+filename+".xml"))
            os.rename(image_path, datasetPath+removedPath+file)

        else:
            xml_path = datasetPath + labelPath + filename+".xml"
            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(xml_path)
            image = cv2.imread(image_path)

            if(labelName is not None):
                i = 0
                for label in labelName:
                    cv2.imshow("Image", imutils.resize(image, width=700))
                    cv2.rectangle(image, (labelXmin[i], labelYmin[i]), (labelXmax[i], labelYmax[i]), (0,255,0), 2)
                    k = cv2.waitKey(1)
                    i += 1

            else:
                print("Moved the image with no labels to ", datasetPath+removed)
                os.rename(image_path, datasetPath+removedPath+"images/"+file)

