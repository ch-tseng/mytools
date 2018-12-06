# -*- coding: utf-8 -*-

import cv2
from imutils.face_utils import rect_to_bb
import dlib
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom

#-------------------------------------------

datasetPath = "/media/sf_VMshare/testlandmark/"
imgPath = "images/"
labelPath = "labels/"
removedPath = "notok/"
okPath = "ok/"

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

    if not os.path.exists(datasetPath+okPath):
        os.makedirs(datasetPath+okPath)
        os.makedirs(datasetPath+okPath+"images")
        os.makedirs(datasetPath+okPath+"labels")
        print("Create the path:", datasetPath + okPath)


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

#--------------------------------------------

chkEnv()

i = 0
imageFolder = datasetPath + imgPath

for file in os.listdir(imageFolder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        print("Processing: ", imageFolder + "/" + file)

        if not os.path.exists(datasetPath+labelPath+filename+".xml"):
            print("Cannot find the file {}, remove this.".format(datasetPath+labelPath+filename+".xml"))

        else:
            image_path = imageFolder + "/" + file
            xml_path = datasetPath + labelPath + filename+".xml"
            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(image_path, xml_path)

            i = 0
            image = cv2.imread(image_path)
            for label in labelName:
                cv2.rectangle(image, (labelXmin[i], labelYmin[i]), (labelXmax[i], labelYmax[i]), (0,255,0), 2)
                i += 1

            cv2.imshow("Image", imutils.resize(image, width=700))
            k = cv2.waitKey(0)

            if(k==110):
                os.rename(image_path, datasetPath+removedPath+"images/"+file)
                os.rename(xml_path, datasetPath+removedPath+"labels/"+filename+".xml")
                print("Moved the image and xml to ", datasetPath+removedPath)
            elif(k==121):
                os.rename(image_path, datasetPath+okPath+"images/"+file)
                os.rename(xml_path, datasetPath+okPath+"labels/"+filename+".xml")
                print("Moved the image and xml to ", datasetPath+okPath)
