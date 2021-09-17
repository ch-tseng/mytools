#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom

#-------------------------------------------

width_resize = 960  #w
datasetPath = r"L:\Dataset\Mine\Plants\pepper_teach"
imgPath = "images/"
labelPath = "labels/"
outputFolder = "resized/"

xml_samplefile = "xml_file.txt"
object_xml_file = "xml_object.txt"

#-------------------------------------------
datasetPath = datasetPath.replace('\\', '/')
xml_path = os.path.join(datasetPath, labelPath)
img_path = os.path.join(datasetPath, imgPath)
out_path = os.path.join(datasetPath, outputFolder)

def chkEnv():
    if not os.path.exists(datasetPath):
        print("There is no such path for dataset:", datasetPath)
        quit()

    if not os.path.exists(xml_path):
        print("There is no such image path for dataset:", xml_path)
        quit()

    if not os.path.exists(img_path):
        print("There is no such image path for dataset:", img_path)
        quit()

    if not os.path.exists(out_path):
        print("create output path of ", out_path)
        os.makedirs(out_path)

    if not os.path.exists(out_path + imgPath):
        print("create output path of ", out_path + imgPath)
        os.makedirs(out_path + imgPath)

    if not os.path.exists(out_path + labelPath):
        print("create output path of ", out_path + labelPath)
        os.makedirs(out_path + labelPath)


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

    return labelName, labelXmin, labelYmin, labelXmax, labelYmax

def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[3]))

    return file_updated

def generateXML(img, file_name, fullpath, bboxes):
    xmlObject = ""
    print("BBOXES:", bboxes)

    (labelName, labelXmin, labelYmin, labelXmax, labelYmax) = bboxes
    for id in range(0, len(labelName)):
        xmlObject = xmlObject + writeObjects(labelName[id], (labelXmin[id], labelYmin[id], labelXmax[id], labelYmax[id]))

    with open(xml_samplefile) as file:
        xmlfile = file.read()

    (h, w, ch) = img.shape
    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", file_name )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + file_name )
    xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

    return xmlfile

def makeDatasetFile(img, img_filename, bboxes):
    file_name, file_ext = os.path.splitext(img_filename)
    jpgFilename = file_name + ".jpg"
    xmlFilename = file_name + ".xml"

    cv2.imwrite(out_path + imgPath + jpgFilename, img)
    print("write to -->", out_path + imgPath + jpgFilename)

    xmlContent = generateXML(img, xmlFilename, out_path + labelPath + xmlFilename, bboxes)
    file = open(out_path + labelPath + xmlFilename, "w")
    file.write(xmlContent)
    file.close
    print("write to -->", out_path + labelPath + xmlFilename)


#-------------------------------------------------------------------

chkEnv()

for file in os.listdir(img_path):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        img_file = file
        xml_file = filename + ".xml"
        img_path = os.path.join(datasetPath, imgPath, img_file)
        xml_path = os.path.join(datasetPath, labelPath, xml_file)
        #print("Processing: ", img_path)

        if(os.path.exists(xml_path)):
            image = cv2.imread(img_path)
            (org_width, org_height) = (image.shape[1], image.shape[0])

            if(org_width>width_resize):
                image = imutils.resize(image, width=width_resize)
            elif(org_height>width_resize):
                image = imutils.resize(image, height=width_resize)

            (img_width, img_height) = (image.shape[1], image.shape[0])
            ratio_w, ratio_h = img_width/org_width, img_height/org_height

            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(xml_path)

            for id in range(0, len(labelXmin)):
                labelXmin[id] = int(labelXmin[id] * ratio_w)
                labelYmin[id] = int(labelYmin[id] * ratio_h)
                labelXmax[id] = int(labelXmax[id] * ratio_w)
                labelYmax[id] = int(labelYmax[id] * ratio_h)

            makeDatasetFile(image, img_file, (labelName, labelXmin, labelYmin, labelXmax, labelYmax))

        else:
            print("Not exists:", xml_path)
