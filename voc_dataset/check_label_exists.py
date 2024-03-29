#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
import shutil
from xml.dom import minidom

#-------------------------------------------
check_labels = ["plate"]
no_label_move_to = r"T:\Datasets\CH_custom\VOC\Vehicles\car_plate_recog_2022\dataset\nolabel"
dataset_images = r"T:\Datasets\CH_custom\VOC\Vehicles\car_plate_recog_2022\dataset\images"
dataset_labels = r"T:\Datasets\CH_custom\VOC\Vehicles\car_plate_recog_2022\dataset\labels"
resize_to = None  #(32, 32)
label_requires = []
#label_requires = [ 'D21', 'D41', 'D51', 'D40' ]  #[] = all
#label_requires = ['D20', 'D21']
crop_add_padding = 0   #add % padding to the copped image

#folderCharacter = "/"  # \\ is for windows
xml_file = "../auto_label_voc/xml_file.txt"
object_xml_file = "../auto_label_voc/xml_object.txt"
#-------------------------------------------
no_label_move_to = no_label_move_to.replace('\\', '/')
dataset_images = dataset_images.replace('\\', '/')
dataset_labels = dataset_labels.replace('\\', '/')

def chkEnv():
    if not os.path.exists(no_label_move_to):
        os.makedirs(no_label_move_to)
        print("no {} folder, created.".format(no_label_move_to))

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

for file in os.listdir(dataset_images):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        print("Processing: ", os.path.join(dataset_images, file))

        if not os.path.exists(os.path.join(dataset_labels, filename+".xml")):
            print("Cannot find the file {} for the image.".format(os.path.join(dataset_labels, filename+".xml")))

        else:
            image_path = os.path.join(dataset_images, file)
            xml_path = os.path.join(dataset_labels, filename+".xml")
            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(image_path, xml_path)

            orgImage = cv2.imread(image_path)
            try:
                test = orgImage.shape

            except:
                continue
                
            image = orgImage.copy()
            for id, label_check in enumerate(check_labels):
                if not label_check in labelName:
                    shutil.copy(image_path, os.path.join(no_label_move_to,file))
                    cv2.imshow("Image", imutils.resize(image, width=700))
                    k = cv2.waitKey(1)


