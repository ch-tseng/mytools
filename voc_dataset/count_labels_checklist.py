#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom
from tqdm import tqdm
import codecs

#-------------------------------------------

report_save_path = "F:/Datasets/MIS_Office/misoffice-more/check_list.txt"
dataset_images = "F:/Datasets/MIS_Office/misoffice-more/images"
dataset_labels = "F:/Datasets/MIS_Office/misoffice-more/labels"

xml_file = "../auto_label_voc/xml_file.txt"
object_xml_file = "../auto_label_voc/xml_object.txt"
#-------------------------------------------

def chkEnv():
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

#--------------------------------------------

chkEnv()

i = 0
line = '{:<7s} {:<42s} {:<5s} {:<300s}'.format('ID', '圖檔名稱', '標記數量', '標記列表')

f = codecs.open(report_save_path, "w", "utf-8")
f.write(line)

for file_id, file in tqdm(enumerate(os.listdir(dataset_images))):
#for file_id, file in enumerate(os.listdir(dataset_images)):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        line = ''
        #print("Processing: ", os.path.join(dataset_images, file))

        if not os.path.exists(os.path.join(dataset_labels, filename+".xml")):
            print("Cannot find the file {} for the image.".format(os.path.join(dataset_labels, filename+".xml")))

        else:
            image_path = os.path.join(dataset_images, file)
            xml_path = os.path.join(dataset_labels, filename+".xml")
            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(image_path, xml_path)

            if(len(labelName)>0):
                names_list = ''
                for id, lname in enumerate(labelName):
                    if(id>0): 
                        names_list += ', '+lname
                    else:
                        names_list += lname

                line = line + '\n' + '{:<7s} {:<50s} {:<5s} {:<300s}'.format(' '+str(file_id)+')', file, str(len(labelName)), names_list)
                f.write(line)

f.close()
