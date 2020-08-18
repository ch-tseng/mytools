#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom

#-------------------------------------------

org_LABEL_NAME = "*"  #*--> all labels
new_LABEL_NAME = "food"  #rename to new label name

dataset_images = "/DATA1/Datasets_projects/Diabnext/images/"
dataset_labels = "/DATA1/Datasets_projects/Diabnext/labels/"

out_path = "/DATA1/Datasets_projects/Diabnext_only_food/"
imgPath = "images/"
labelPath = "labels/"

folderCharacter = "/"  # \\ is for windows
xml_samplefile = "xml_file.txt"
object_xml_file = "xml_object.txt"

#-------------------------------------------

def chkEnv():
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print("no {} folder, created.".format(out_path))

    if not os.path.exists(out_path + folderCharacter + imgPath):
        os.makedirs(out_path + folderCharacter + imgPath)
        print("no {} folder, created.".format(out_path + folderCharacter + imgPath))

    if not os.path.exists(out_path + folderCharacter + labelPath):
        os.makedirs(out_path + folderCharacter + labelPath)
        print("no {} folder, created.".format(out_path + folderCharacter + labelPath))


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

    #cv2.imwrite(out_path + imgPath + jpgFilename, img)
    #print("write to -->", out_path + imgPath + jpgFilename)

    xmlContent = generateXML(img, xmlFilename, out_path + labelPath + xmlFilename, bboxes)
    file = open(out_path + labelPath + xmlFilename, "w")
    file.write(xmlContent)
    file.close
    print("write to -->", out_path + labelPath + xmlFilename)

#--------------------------------------------

chkEnv()

i = 0

for file in os.listdir(dataset_images):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        print("Processing: ", dataset_images + file)

        if not os.path.exists(dataset_labels+filename+".xml"):
            print("Cannot find the file {} for the image.".format(dataset_labels+filename+".xml"))

        else:
            image_path = dataset_images + file
            xml_path = dataset_labels + filename+".xml"
            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(image_path, xml_path)

            for id, label in enumerate(labelName):
                if(org_LABEL_NAME=='*'):
                    labelName[id] = new_LABEL_NAME
                else:
                    if(label==org_LABEL_NAME):
                        labelName[id] = new_LABEL_NAME

            makeDatasetFile(cv2.imread(image_path), file, (labelName, labelXmin, labelYmin, labelXmax, labelYmax))
