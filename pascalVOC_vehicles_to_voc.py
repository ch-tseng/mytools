# -*- coding: utf-8 -*-

import cv2
from shutil import copyfile
import os, time
import os.path
from xml.dom import minidom

#-------------------------------------------

labels_want = ["car", "truck", "bus", "motorbike", "bicycle", "person"]
pose_label = False
pose = ["rear", "left", "right", "frontal"]  #use lower case, []--> all, ["rear", "left", "right", "frontal", "unspecified"]

source_voc_path = "/DATA1/Datasets_download/Labeled/VOC/VOC_Dataset/2012/VOCdevkit/VOC2012/"
source_images = "JPEGImages"
source_labels = "Annotations"

target_voc_path = "/DATA1/Datasets_mine/labeled/sunplusit_official_cars"
target_images = "images/"
target_labels = "labels/"
imgType = "jpg"
xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"


def chkEnv():
    if not os.path.exists(os.path.join(source_voc_path, source_images)):
        print("There is no source dataset in this path:", os.path.join(source_voc_path, source_images))
        quit()

    if not os.path.exists(os.path.join(source_voc_path, source_labels)):
        print("There is no source dataset in this path:", os.path.join(source_voc_path, source_labels))
        quit()

    if not os.path.exists(os.path.join(target_voc_path, target_images)):
        os.makedirs(os.path.join(target_voc_path, target_images))
        print("Create the path:", os.path.join(target_voc_path, target_images))

    if not os.path.exists(os.path.join(target_voc_path, target_labels)):
        os.makedirs(os.path.join(target_voc_path, target_labels))
        print("Create the path:", os.path.join(target_voc_path, target_labels))

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

    #print(xmlFile)
    objects = labelXML.getElementsByTagName("object")

    for object in objects:
        pose_list = []
        if(pose_label is True):
            tmpArrays = object.getElementsByTagName("pose")
            for id, elem in enumerate(tmpArrays):
                pose_list.append(elem.firstChild.data)

        id_list = []
        tmpArrays = object.getElementsByTagName("name")
        for id, elem in enumerate(tmpArrays):
            if(str(elem.firstChild.data).lower() in labels_want):
                id_list.append(id)
                #print(xmlFile, id, pose_list, "TEST:", pose_list[id].lower())
                if(len(pose_list)>id):
                    if((pose_list[id].lower() in pose) and len(pose)>0):
                        labelName.append(str(elem.firstChild.data) + "_" + pose_list[id])
                else:
                    labelName.append(str(elem.firstChild.data))

            tmpArrays = object.getElementsByTagName("xmin")
            for id, elem in enumerate(tmpArrays):
                if(id in id_list):
                    labelXmin.append(int(float(elem.firstChild.data)))

            tmpArrays = object.getElementsByTagName("ymin")
            for id, elem in enumerate(tmpArrays):
                if(id in id_list):
                    labelYmin.append(int(float(elem.firstChild.data)))

            tmpArrays = object.getElementsByTagName("xmax")
            for id, elem in enumerate(tmpArrays):
                if(id in id_list):
                    labelXmax.append(int(float(elem.firstChild.data)))

            tmpArrays = object.getElementsByTagName("ymax")
            for id, elem in enumerate(tmpArrays):
                if(id in id_list):
                    labelYmax.append(int(float(elem.firstChild.data)))

    return labelName, labelXmin, labelYmin, labelXmax, labelYmax

def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[0] + bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[1] + bbox[3]))

    return file_updated

def generateXML(imgfile, filename, fullpath, bboxes, imgfilename):
    xmlObject = ""

    for (labelName, bbox) in bboxes:
        #for bbox in bbox_array:
        xmlObject = xmlObject + writeObjects(labelName, bbox)

    with open(xml_file) as file:
        xmlfile = file.read()

    img = cv2.imread(imgfile)
    #print(os.path.join(datasetPath, imgPath, imgfilename))
    cv2.imwrite(os.path.join(target_voc_path, target_images, imgfilename), img)

    (h, w, ch) = img.shape
    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", filename )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + filename )
    xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

    return xmlfile

def makeLabelFile(filename, bboxes, imgfile):
    jpgFilename = filename + "." + imgType
    xmlFilename = filename + ".xml"

    #cv2.imwrite(os.path.join(datasetPath, imgPath, jpgFilename), img)

    xmlContent = generateXML(imgfile, xmlFilename, os.path.join(target_voc_path, target_labels, xmlFilename), bboxes, jpgFilename)
    file = open(os.path.join(target_voc_path, target_labels, xmlFilename), "w")
    file.write(xmlContent)
    file.close


#--------------------------------------------
if __name__ == "__main__":
    chkEnv()

    i = 0
    imageFolder = os.path.join(source_voc_path, source_images)


    for file in os.listdir(imageFolder):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            #print("Processing: ", imageFolder + "/" + file)

            xml_path = os.path.join(source_voc_path, source_labels, filename+".xml")
            
            if os.path.exists(xml_path):
                image_path = os.path.join(imageFolder, file)
                labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(image_path, xml_path)
                img_bboxes = []
                print(labelName, labelXmin, labelYmin, labelXmax, labelYmax)
                for i, label_want in enumerate(labelName):
                    x = int(float(labelXmin[i]))
                    y = int(float(labelYmin[i]))
                    w = int(float(labelXmax[i]))-int(float(labelXmin[i]))
                    h = int(float(labelYmax[i]))-int(float(labelYmin[i]))
                    img_bboxes.append( (label_want, [x,y,w,h])  )

                if(len(img_bboxes)>0):
                    print(img_bboxes)
                    makeLabelFile(filename, img_bboxes, image_path)
