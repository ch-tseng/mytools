#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom

#-------------------------------------------

want_LABEL_NAME = ["person_head", "person_vbox"]  #rename to new label name

#dataset_images = "F:/Datasets/crowd_human_official_voc/images"
#dataset_labels = "F:/Datasets/crowd_human_official_voc/labels"
dataset_images = "F:/Datasets/head_body_Eden/images"
dataset_labels = "F:/Datasets/head_body_Eden/labels"

out_path = "F:/Datasets/crowndHuman_2_classes"
imgPath = "images/"
labelPath = "labels/"

xml_samplefile = "xml_file.txt"
object_xml_file = "xml_object.txt"

#-------------------------------------------

def chkEnv():
    if not os.path.exists(out_path):
        os.makedirs(out_path)
        print("no {} folder, created.".format(out_path))

    if not os.path.exists(os.path.join(out_path, imgPath)):
        os.makedirs(os.path.join(out_path, imgPath))
        print("no {} folder, created.".format(os.path.join(out_path, imgPath)))

    if not os.path.exists(os.path.join(out_path, labelPath)):
        os.makedirs(os.path.join(out_path, labelPath))
        print("no {} folder, created.".format(os.path.join(out_path, labelPath)))


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

    print(imgFile)
    try:
        (h,w,c) = cv2.imread(imgFile).shape
    except:
        print("Erro file")
        return None, None, None, None, None

    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        labelName.append(str(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmin")
    for elem in tmpArrays:
        v = int(elem.firstChild.data)
        if(v<0): v=0
        labelXmin.append(v)

    tmpArrays = labelXML.getElementsByTagName("ymin")
    for elem in tmpArrays:
        v = int(elem.firstChild.data)
        if(v<0): v=0
        labelYmin.append(v)

    tmpArrays = labelXML.getElementsByTagName("xmax")
    for elem in tmpArrays:
        v = int(elem.firstChild.data)
        if(v<0): v=0
        if(v>w): v=w
        labelXmax.append(v)

    tmpArrays = labelXML.getElementsByTagName("ymax")
    for elem in tmpArrays:
        v = int(elem.firstChild.data)
        if(v<0): v=0
        if(v>h): v=h
        labelYmax.append(v)

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

def makeDatasetFile(img, img_filename, bboxes):
    file_name, file_ext = os.path.splitext(img_filename)
    jpgFilename = file_name + ".jpg"
    xmlFilename = file_name + ".xml"

    #cv2.imwrite(out_path + imgPath + jpgFilename, img)
    #print("write to -->", out_path + imgPath + jpgFilename)

    xmlContent = generateXML(img, xmlFilename, os.path.join(out_path, labelPath, xmlFilename), bboxes)
    file = open(os.path.join(out_path, labelPath, xmlFilename), "w")
    file.write(xmlContent)
    file.close
    #print("write to -->", os.path.join(out_path, labelPath, xmlFilename))

#--------------------------------------------

chkEnv()

i = 0

for file in os.listdir(dataset_images):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        #print("Processing: ", os.path.join(dataset_images, file))

        if not os.path.exists(os.path.join(dataset_labels, filename+".xml")):
            print("Cannot find the file {} for the image.".format(os.path.join(dataset_labels, filename+".xml")))

        else:
            image_path = os.path.join(dataset_images, file)
            #image_path = dataset_images + '/' +  file
            xml_path = os.path.join(dataset_labels, filename+".xml")
            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(image_path, xml_path)

            if(labelName is not None):
                n_labels, n_xmin, n_ymin, n_xmax, n_ymax = [], [], [], [], []
                for id, label in enumerate(labelName):
                    if(label in want_LABEL_NAME):
                        if(labelXmax[id]>labelXmin[id] and labelYmax[id]>labelYmin[id]):
                            n_labels.append(label)
                            n_xmin.append(labelXmin[id])
                            n_ymin.append(labelYmin[id])
                            n_xmax.append(labelXmax[id])
                            n_ymax.append(labelYmax[id])
                            #print(label, "add to dataset")
                        else:
                            print("BBOX is not valid:", (labelXmin[id], labelYmin[id], labelXmax[id], labelYmax[id] ))

                if(len(n_labels)>0):
                    makeDatasetFile(cv2.imread(image_path), file, (n_labels, n_xmin, n_ymin, n_xmax, n_ymax))