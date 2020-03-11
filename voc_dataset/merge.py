# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time, glob
import os.path
import numpy as np
from xml.dom import minidom

#------------------------------------------------------------------------------------------------------
labelFolders = [ "Test-0/labels_D00", "Test-0/labels_D01", "Test-0/labels_D10", "Test-0/labels_D11", \
    "Test-0/labels_D20", "Test-0/labels_D21", "Test-0/labels_D40", "Test-0/labels_D42", "Test-0/labels_D91"]
target_label_folder = "Test-0/labels"
target_image_folder = "Test-0/images"

#------------------------------------------------------------------------------------------------------
xml_samplefile = "xml_file.txt"
object_xml_file = "xml_object.txt"

def get_sorted_files(file_path, ftype):
    files = list(filter(os.path.isfile, glob.glob(file_path + "/*."+ftype)))
    return files

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

def makeDatasetFile(xml_filename, bboxes):
    file_name, file_ext = os.path.splitext(xml_filename)
    jpgFilename = file_name + ".jpg"
    xmlFilename = file_name + ".xml"

    if(os.path.exists(os.path.join(target_image_folder, jpgFilename))):
        img = cv2.imread(os.path.join(target_image_folder, jpgFilename))
        xmlContent = generateXML(img, xmlFilename, os.path.join(target_label_folder, xmlFilename), bboxes)
        file = open(os.path.join(target_label_folder, xmlFilename), "w")
        file.write(xmlContent)
        file.close
        print("write to -->", os.path.join(target_label_folder, xmlFilename))            
    else:
        print("Error, cannot find image:", os.path.join(target_image_folder, jpgFilename))

if __name__ == '__main__':

    for waiting_folder in labelFolders:
        print("processing", waiting_folder)

        for id, xml_file in enumerate(get_sorted_files(waiting_folder, "xml")):
            print("    {} {}".format(id+1, xml_file))
            xml_basename = os.path.basename(xml_file)
            target_xml = os.path.join(target_label_folder, xml_basename)

            labelName, labelXmin, labelYmin, labelXmax, labelYmax = [], [], [], [], []
            if (os.path.exists(target_xml)):
                labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(target_xml)

            new_labelName, new_labelXmin, new_labelYmin, new_labelXmax, new_labelYmax = [], [], [], [], []
            new_labelName, new_labelXmin, new_labelYmin, new_labelXmax, new_labelYmax = getLabels(xml_file)

            if(len(new_labelName)>0):
                new_labelName += labelName
                new_labelXmin += labelXmin
                new_labelYmin += labelYmin
                new_labelXmax += labelXmax
                new_labelYmax += labelYmax

                makeDatasetFile(xml_basename, (new_labelName, new_labelXmin, new_labelYmin, new_labelXmax, new_labelYmax))