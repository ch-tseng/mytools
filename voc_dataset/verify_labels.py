#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os, sys
import numpy as np
from xml.dom import minidom

verify_labels = { "leye":3, "reye":3, "lmouth":2, "rmouth":2 }
target_xml_path = "/home/digits/tmp/veri_labels/labels/"


#-------------------------------------------------------------------------
def chkEnvironment():
    if(not os.path.exists(target_xml_path)):
        print("No such folder:", target_xml_path)
        sys.exit(1)

    if(len(verify_labels)<1):
        print("Please put some labesl to the [verify_labels] parameter")
        sys.exit(1)

def fileCount(path, ftype):
    i = 0
    for file in os.listdir(path):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(ftype=="images"):
            if(file_extension in [".jpg", ".jpeg", ".png", ".bmp", "pcx"]):
                i += 1
        else:
            if(file_extension == ftype):
                i += 1

    return i

def dict_compare(d1, d2):
    d1_keys = set(d1.keys())
    d2_keys = set(d2.keys())
    intersect_keys = d1_keys.intersection(d2_keys)
    added = d1_keys - d2_keys
    removed = d2_keys - d1_keys
    modified = {o : (d1[o], d2[o]) for o in intersect_keys if d1[o] != d2[o]}
    same = set(o for o in intersect_keys if d1[o] == d2[o])

    return added, removed, modified, same

def countLabels(xmlFile):
    if(not os.path.isfile(xmlFile)):
        return 0

    #filename, file_extension = os.path.splitext(xmlfile)
    labelXML = minidom.parse(xmlFile)
    labelNames = {}

    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        label_name = str(elem.firstChild.data)
        #print("label_name", label_name)
        if(label_name in labelNames):
            count = labelNames[label_name] + 1
        else:
            count = 1

        update_dict = {label_name:count}
        #print(label_name, "-->", update_dict)

        labelNames.update(update_dict)

    #print("Final: ", labelNames)

    added, removed, modified, same = dict_compare(labelNames, verify_labels)

    rtnResult = True
    if(len(added)>0 or len(removed)>0 or len(modified)>0):
        rtnResult = False
        print('')
        print(xmlfile)

    if(len(added)>0):
        print("  標記名稱有多:", added)

    if(len(removed)>0):
        print("  標記名稱有缺:", removed)

    if(len(modified)>0):
        print("  標記數目不對:", modified)

    #print("same:", same)

    return rtnResult

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

#--------------------------------------------------------------------------
if __name__ == '__main__':

    labelCount = []
    chkEnvironment()
    errCount = 0

    for xmlfile in os.listdir(target_xml_path):
        filename, file_extension = os.path.splitext(xmlfile)
        file_extension = file_extension.lower()

        if(file_extension == '.xml'):
            label_check = countLabels(os.path.join(target_xml_path, xmlfile))

            if(label_check is False):
                errCount += 1

    print('')
    print("有問題的XML檔案數: ", errCount)
