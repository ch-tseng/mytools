#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import os, sys
import numpy as np
from xml.dom import minidom

sources = []

sources.append("/media/sf_datasets/temp/orange-v1")
sources.append("/media/sf_datasets/temp/orange-v2")
sources.append("/media/sf_datasets/temp/orange-v3")


img_folder = "/media/sf_datasets/temp/orange-v1/images/"
lbl_folder = "labels"
img_type = "images"  #all types of images
lbl_type = ".xml"

th_position = 5
th_size = 60
#---------------------------------------------------------
def chkFolder(path):
    if(os.path.isdir(path)):
        return True
    else:
        return False

def chkEnvironment():
    for source in sources:
        if(chkFolder(source) is False):
            print("[error]: source: {} is not exists".format(source))
            sys.exit()

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

def countLabels(xmlFile):
    if(not os.path.isfile(xmlFile)):
        return 0

    labelXML = minidom.parse(xmlFile)
    labelName = []

    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        labelName.append(str(elem.firstChild.data))

    return len(labelName)

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


#---------------------------------------------------------
errorMSG = []
file_counts = []

if __name__ == '__main__': 
    chkEnvironment()

    err = False
    print('')
    print('[Stage #1] 標記檔案數量比對----------------------------------------------')
    total_lblfile , total_imgfile = 0, 0
    for id, s in enumerate(sources):
        #s_count1 = fileCount(os.path.join(s, img_folder ), img_type)
        s_count1 = fileCount(img_folder, img_type)
        s_count2 = fileCount(os.path.join(s, lbl_folder ), lbl_type)
        file_counts.append( (s_count1, s_count2) )
        total_lblfile += s_count1
        total_imgfile += s_count2
        print("    來源{} --> 圖片檔案數量:{}, 標記檔案數量:{}".format(id+1, s_count1, s_count2) )

    #if(total_imgfile / len(sources) != s_count1):
    #    print("")
    #    print("圖片檔案數量不一致, 請先確認。 ")
    #    err = True
    if(total_lblfile / len(sources) != s_count2):
        print("")
        print("    標記檔案數量不一致, 請先確認。 ")
        err = True

        for id, s in enumerate(sources):
            print("")
            print("    來源{}:".format(id+1) )
            files = []

            for file in os.listdir(img_folder):
                filename, file_extension = os.path.splitext(file)
                file_extension = file_extension.lower()
                if( os.path.isfile(os.path.join(sources[id], lbl_folder, filename+lbl_type) ) is False):
                    files.append(filename+file_extension)

            if(len(files)>0):
                print("        --> 少標記檔案:")
                print("           ", files)
            else:
                print("        --> 正確，標記檔案與圖片檔案一致。 ")

    print('')
    if(err is True):
        sys.exit()

    print('[Stage #2] 標記名稱差異比對----------------------------------------------')
    i = 0
    err = False
    for file in os.listdir(os.path.join(sources[0], lbl_folder)):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == lbl_type):
            labelCount = []

            for id in range(0, len(sources)):
                source_path = os.path.join(sources[id], lbl_folder, file)
                labelCount.append(countLabels(source_path))

            total = 0
            for countNum in labelCount:
                total += countNum

            if(total/len(labelCount) != labelCount[0]):
                i += 1
                print("    {}) 圖檔:{} 標記數量不一致, 分別為:{}".format(i, filename+file_extension, labelCount) )
                err = True

    print("")
    if(err is True):
        print("    標記數量不一致, 請先確認。 ")
        print("")
        sys.exit()

    print('')
    print('[Stage #3] 標記種類差異比對----------------------------------------------')
    i = 0
    err = False
    for file in os.listdir(os.path.join(sources[0], lbl_folder)):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == lbl_type):
            labelList = []

            for id in range(0, len(sources)):
                source_path = os.path.join(sources[id], lbl_folder, file)
                labelNames, _, _, _, _ = getLabels(source_path)
                labelList.append(labelNames)

            ii = 0
            err2 = False
            labelsGroup = []
            for labels in labelList:
                nlabels = sorted(labels)
                if(ii == 0):
                    last_labels = nlabels
                if(nlabels == last_labels) is False:
                    err2 = True

                ii += 1

            if(err2 is True):
                i += 1
                print("")
                print("    {}) 圖檔:{} 標記種類有差異:".format(i, filename+file_extension))
                for id, labels in enumerate(labelList):
                    print("          來源{}. 標記列表:{}: ".format(id, labels) )

                err = True

    if(err is True):
        print("")
        print("    以上{}個圖檔的標記名稱不一致，請先更正。".format(i))
        print("")
        sys.exit()

    print('')
    print('[Stage #4] 標記位置差異----------------------------------------------')
    i = 0
    err = False
    for file in os.listdir(os.path.join(sources[0], lbl_folder)):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == lbl_type):
            labelList = []

            for id in range(0, len(sources)):
                source_path = os.path.join(sources[id], lbl_folder, file)
                _, x1, y1, x2, y2 = getLabels(source_path)
                labels = []
                for ii, x in enumerate(x1):
                    area = math.sqrt(x2[ii]-x1[ii])*(y2[ii]-y1[ii])
                    labels.append((x1[ii]+x2[ii])/2)
                    labels.append((y1[ii]+y2[ii])/2)
                    labels.append(area)

                nlabels = np.array(labels)
                #print(nlabels)
                labelList.append(labels)

            nlabelList = np.array(labelList)
            #print(nlabelList.var(), nlabelList.std())
            try:
                variance = np.average(np.std(nlabelList, axis=0))
            except:
                variance = 999999

            if(variance>th_position):
                print("Variance:", variance)
                i += 1
                print("    {}) 圖檔:{} 標記位置差異大:{}".format(i, filename+file_extension, variance) )
                err = True

    print('')
    print('            標記尺寸差異----------------------------------------------')
    i = 0
    err = False
    for file in os.listdir(os.path.join(sources[0], lbl_folder)):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == lbl_type):
            labelList = []

            for id in range(0, len(sources)):
                source_path = os.path.join(sources[id], lbl_folder, file)
                _, x1, y1, x2, y2 = getLabels(source_path)
                labels = []
                for ii, x in enumerate(x1):
                    area = math.sqrt(x2[ii]-x1[ii])*(y2[ii]-y1[ii])
                    labels.append((x1[ii]+x2[ii])/2)
                    labels.append((y1[ii]+y2[ii])/2)
                    labels.append(area)

                nlabels = np.array(labels)
                #print(nlabels)
                labelList.append(labels)

            nlabelList = np.array(labelList)
            #print(nlabelList.var(), nlabelList.std())

            try:
                variance = np.average(np.std(nlabelList, axis=0))
            except:
                variance = 999999

            if(variance>th_size):
                print("Variance:", variance)
                i += 1
                print("    {}) 圖檔:{} 標記尺寸差異大:{}".format(i, filename+file_extension, variance) )
                err = True

    print("")
