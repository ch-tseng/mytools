#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time, shutil
import os.path
import numpy as np
from xml.dom import minidom

#-------------------------------------------
#extract_to = "H:/Datasets/Weight_Vegetables/extract"
#imgFolder = "H:/Datasets/Weight_Vegetables/images"
#xmlFolder = "H:/Datasets/Weight_Vegetables/labels"
extract_to = "D:/temp/extract"
imgFolder = "M:/Diabnext/dataset/images"
yoloFolder = "M:/Diabnext/dataset/labels_final_20200728"
class_path = "M:/Diabnext/dataset/labels_final_20200728/classes.txt"

target_voc_path = "M:/Diabnext/dataset/labels_final_20200728_voc"
target_labels = "labels"
target_images = "images"
resize_to = None  #(32, 32)
rmove_error_file_to = "M:/Diabnext/dataset/cannot_read_images"
xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#-------------------------------------------

class_list = {}
f = open(class_path, 'r',encoding="utf-8")
line = f.readline()
id = 0
while line:
    line = line.replace('\n','')
    name = line.strip()
    class_list.update( {id:name })
    id += 1
    line = f.readline()
f.close()

print(class_list)

def chkEnv():
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print("no {} folder, created.".format(extract_to))

    if not os.path.exists(rmove_error_file_to):
        os.makedirs(rmove_error_file_to)
        print("no {} folder, created.".format(rmove_error_file_to))

    if(not os.path.exists(imgFolder)):
        print("There is no such folder {}".format(imgFolder))
        quit()

    if(not os.path.exists(yoloFolder)):
        print("There is no such folder {}".format(yoloFolder))
        quit()

    if not os.path.exists(os.path.join(target_voc_path, target_images)):
        os.makedirs(os.path.join(target_voc_path, target_images))
        print("Create the path:", os.path.join(target_voc_path, target_images))

    if not os.path.exists(os.path.join(target_voc_path, target_labels)):
        os.makedirs(os.path.join(target_voc_path, target_labels))
        print("Create the path:", os.path.join(target_voc_path, target_labels))        

def getLabels(img, yoloFile):
    f = open(yoloFile, 'r',encoding="utf-8")
    line = f.readline()

    labelName, labelXmin, labelYmin, labelXmax, labelYmax = [], [], [], [], []
    while line:
        line = line.replace('\n','')
        datas = line.split(' ')
        
        if(len(datas)==5):
            width, height = img.shape[1], img.shape[0]
            x = int(width*float(datas[1]))
            y = int(height*float(datas[2]))
            w = int(width*float(datas[3]))
            h = int(height*float(datas[4]))
            x = int(x - w/2)
            y = int(y - h/2)
            try:
                class_name = class_list[int(datas[0])]
            
                #print("class:{} x:{}, y:{}, w:{}, h:{}".format(class_name, x, y, w, h))

                labelName.append(class_name)
                labelXmin.append(x)
                labelYmin.append(y)
                labelXmax.append(x+w)
                labelYmax.append(y+h)

            except:
                print("Error class name:", datas)

        line = f.readline()

    f.close()

    return labelName, labelXmin, labelYmin, labelXmax, labelYmax

def write_lale_images(label, img, saveto, filename):
    writePath = os.path.join(extract_to,label)
    #print("WRITE:", writePath)

    if not os.path.exists(writePath):
        os.makedirs(writePath)

    if(resize_to is not None):
        img = cv2.resize(img, resize_to)

    try:
        cv2.imwrite(os.path.join(writePath, filename), img)
    except:
        print("write error: ", os.path.join(writePath, filename))

def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[0] + bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[1] + bbox[3]))

    return file_updated

def generateXML(imgfile, img, filename, fullpath, bboxes, imgfilename):
    xmlObject = ""

    for (labelName, bbox) in bboxes:
        #for bbox in bbox_array:
        xmlObject = xmlObject + writeObjects(labelName, bbox)

    with open(xml_file) as file:
        xmlfile = file.read()

    #img = cv2.imread(imgfile)
    #cv2.imwrite(os.path.join(target_voc_path, target_images, imgfilename), img)

    (h, w, ch) = img.shape
    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", filename )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + filename )
    xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

    return xmlfile

def makeLabelFile(filename, img, bboxes, imgfile, imgType):
    jpgFilename = filename + imgType
    xmlFilename = filename + ".xml"

    #cv2.imwrite(os.path.join(datasetPath, imgPath, jpgFilename), img)

    xmlContent = generateXML(imgfile, img, xmlFilename, os.path.join(target_voc_path, target_labels, xmlFilename), bboxes, jpgFilename)
    file = open(os.path.join(target_voc_path, target_labels, xmlFilename), "w", encoding="utf-8")
    file.write(xmlContent)
    file.close

#--------------------------------------------

chkEnv()

i = 0

for fid, file in enumerate(os.listdir(imgFolder)):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        #print("Processing: ", os.path.join(imgFolder, file))

        if not os.path.exists(os.path.join(yoloFolder, filename+".txt")):
            print("Cannot find the file {} for the image.".format(os.path.join(yoloFolder, filename+".txt")))

        else:
            image_path = os.path.join(imgFolder, file)
            yolo_path = os.path.join(yoloFolder, filename+".txt")
            orgImage = cv2.imread(image_path)
            try:
                test = orgImage.shape
            except:
                print(fid, "Error read:", image_path)
                shutil.move(image_path, os.path.join(rmove_error_file_to, file))
                continue

            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(orgImage, yolo_path)
            #print(orgImage.shape)
            #print(labelName, labelXmin, labelYmin, labelXmax, labelYmax)

            #image = orgImage.copy()
            #for id, label in enumerate(labelName):
            #    cv2.rectangle(image, (labelXmin[id], labelYmin[id]), (labelXmax[id], labelYmax[id]), (0,255,0), 2)
            #    label_area = orgImage[labelYmin[id]:labelYmax[id], labelXmin[id]:labelXmax[id]]
            #    label_img_filename = filename + "_" + str(id) + ".jpg"
            #    write_lale_images(labelName[id], label_area, extract_to, label_img_filename)

            #cv2.imshow("Image", imutils.resize(image, width=700))
            #k = cv2.waitKey(1)
            img_bboxes = []
            print(labelName, labelXmin, labelYmin, labelXmax, labelYmax)
            for i, label_want in enumerate(labelName):
                x = int(float(labelXmin[i]))
                y = int(float(labelYmin[i]))
                w = int(float(labelXmax[i]))-int(float(labelXmin[i]))
                h = int(float(labelYmax[i]))-int(float(labelYmin[i]))
                img_bboxes.append( (label_want, [x,y,w,h])  )

            if(len(img_bboxes)>0):    
                makeLabelFile(filename, orgImage, img_bboxes, image_path, file_extension)
