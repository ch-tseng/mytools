#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom

#-------------------------------------------

#org_LABEL_NAME = "minibuss"  #*--> all labels
#new_LABEL_NAME = "bus"  #rename to new label name
LABEL_NAME_UPDATE = { "I":"1", "O":"0", "Z":"2", "B":"8" }

dataset_images = "D:/works/car_plate_easyocr/img_plates"
dataset_labels = "D:/works/car_plate_easyocr/img_labels"

out_path = "D:/works/car_plate_chars"
imgPath = "images/"
labelPath = "labels/"

xml_samplefile = "xml_file.txt"
object_xml_file = "xml_object.txt"

#-------------------------------------------

def chkEnv():
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print("no {} folder, created.".format(out_path))

    if not os.path.exists(os.path.join(out_path ,imgPath)):
        os.makedirs(os.path.join(out_path ,imgPath))
        print("no {} folder, created.".format(os.path.join(out_path ,imgPath)))

    if not os.path.exists(os.path.join(out_path ,labelPath)):
        os.makedirs(os.path.join(out_path ,labelPath))
        print("no {} folder, created.".format(os.path.join(out_path ,labelPath)))


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
    #print("BBOXES:", bboxes)

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

def makeDatasetFile(img, img_filename, bboxes, id_file):
    #file_name, file_ext = os.path.splitext(img_filename)
    file_name = str(id_file).zfill(8)
    jpgFilename = file_name + ".jpg"
    xmlFilename = file_name + ".xml"

    cv2.imwrite(os.path.join(out_path,imgPath,jpgFilename), img)
    #print("write to -->", os.path.join(out_path,imgPath,jpgFilename))

    xmlContent = generateXML(img, os.path.join(xmlFilename, out_path, labelPath), xmlFilename, bboxes)
    file = open( os.path.join(out_path, labelPath, xmlFilename), "w")
    file.write(xmlContent)
    file.close
    #print("write to -->", os.path.join(out_path, labelPath, xmlFilename))

#--------------------------------------------

chkEnv()

id_file = 0
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

            for id, label in enumerate(labelName):
                for oid, old_label in enumerate(LABEL_NAME_UPDATE):
                    if(label.lower()==old_label.lower()):
                        labelName[id] = LABEL_NAME_UPDATE[old_label]

            err = False
            try:
                img_data = cv2.imread(image_path)
                shape = img_data.shape
            except:
                print("Image", image_path, "cannot read." )
                err = True

            if(err is False):
                id_file += 1
                makeDatasetFile(img_data, file, (labelName, labelXmin, labelYmin, labelXmax, labelYmax), id_file)
