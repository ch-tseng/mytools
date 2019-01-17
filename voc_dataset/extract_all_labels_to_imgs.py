#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom

#-------------------------------------------

extract_to = "/media/sf_datasets/temp/breads/"
dataset_images = "/media/sf_datasets/categories/breads_POS/images/"
dataset_labels = "/media/sf_datasets/categories/breads_POS/labels/"

folderCharacter = "/"  # \\ is for windows
xml_file = "../auto_label_voc/xml_file.txt"
object_xml_file = "../auto_label_voc/xml_object.txt"
#-------------------------------------------

def chkEnv():
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print("no {} folder, created.".format(extract_to))

    if(not os.path.exists(dataset_images)):
        print("There is no such folder {}".format(dataset_images))
        quit()

    if(not os.path.exists(dataset_labels)):
        print("There is no such folder {}".format(dataset_labels))
        quit()

    if(not os.path.exists(xml_file)):
        print("There is no xml file in {}".format(xml_file))
        quit()

    if(not os.path.exists(object_xml_file)):
        print("There is no object xml file in {}".format(object_xml_file))
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
    writePath = extract_to+label
    print("WRITE:", writePath)

    if not os.path.exists(writePath):
        os.makedirs(writePath)

    cv2.imwrite(writePath + folderCharacter + filename, img)

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

            orgImage = cv2.imread(image_path)
            image = orgImage.copy()
            for id, label in enumerate(labelName):
                cv2.rectangle(image, (labelXmin[id], labelYmin[id]), (labelXmax[id], labelYmax[id]), (0,255,0), 2)
                label_area = orgImage[labelYmin[id]:labelYmax[id], labelXmin[id]:labelXmax[id]]
                label_img_filename = filename + "_" + str(id) + ".jpg"
                write_lale_images(labelName[id], label_area, extract_to, label_img_filename)

            cv2.imshow("Image", imutils.resize(image, width=700))
            k = cv2.waitKey(1)

