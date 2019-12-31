# -*- coding: utf-8 -*-

import cv2
from shutil import copyfile
import os, time
import os.path
from xml.dom import minidom

#-------------------------------------------

source_xml_landmarks_path = "/DATA1/Datasets_download/Labeled/VOC/hand_dataset/HGR/HGR1/feature_points/"
source_images = "/DATA1/Datasets_download/Labeled/VOC/hand_dataset/HGR/HGR1/original_images/"

target_ds_path = "/DATA1/Datasets_download/Labeled/VOC/hand_dataset/HGR_json"
target_images = "images/"
target_labels = "jsons/"
imgType = ".jpg"
xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

def chkEnv():
    if not os.path.exists(source_xml_landmarks_path):
        print("There is no landmarks dataset in this path:", source_xml_landmarks_path)
        quit()

    if not os.path.exists(source_images):
        print("There is no images in this path:", source_images)
        quit()

    if not os.path.exists(os.path.join(target_ds_path, target_images)):
        os.makedirs(os.path.join(target_ds_path, target_images))
        print("Create the path:", os.path.join(target_ds_path, target_images))

    if not os.path.exists(os.path.join(target_ds_path, target_labels)):
        os.makedirs(os.path.join(target_ds_path, target_labels))
        print("Create the path:", os.path.join(target_ds_path, target_labels))

def getLabels(imgFile, xmlFile):
    labelXML = minidom.parse(xmlFile)
    points = []
    categories = []
    types = []

    #print(labelXML)
    objects = labelXML.getElementsByTagName("hand")

    for object in objects:
        features = object.getElementsByTagName("FeaturePoint")
        if(not len(features)>0):
            continue

        for feature in features:
            x_mark = int(feature.getAttribute("x"))
            y_mark = int(feature.getAttribute("y"))
            cate_mark = feature.getAttribute("category")
            type_mark = feature.getAttribute("type")
            points.append((x_mark, y_mark))
            categories.append(cate_mark)
            types.append(type_mark)

    return types, categories, points
    
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

def makeLabelFile(jsonFilename, json_list, imgfile, img):
    cv2.imwrite(imgfile, img)

    file = open(jsonFilename, "w")
    for line in json_list:
        file.write(line + '\n')
    file.close


#--------------------------------------------
if __name__ == "__main__":
    chkEnv()
    #type_points, cate_points, points = getLabels("/DATA1/Datasets_download/Labeled/VOC/hand_dataset/HGR/HGR1/original_images/T_P_hgr1_id03_7.jpg", \
    #    "/DATA1/Datasets_download/Labeled/VOC/hand_dataset/HGR/HGR1/feature_points/1_P_hgr1_id02_1.xml")
    

    i = 0
    imageFolder = source_images

    for file in os.listdir(imageFolder):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            #print("Processing: ", imageFolder + "/" + file)

            xml_path = os.path.join(source_xml_landmarks_path, filename+".xml")
            img_filename = filename + imgType
            json_filename = filename + '.json'

            if os.path.exists(xml_path):
                image_path = os.path.join(imageFolder, file)
                type_points, cate_points, points = getLabels(image_path, xml_path)
                try:
                    img = cv2.imread(image_path)
                    (img_h, img_w, _) = img.shape
                except:
                    continue

                point_list = []
                print(type_points, cate_points, points)

                for i, point in enumerate(points):
                    x = point[0]
                    y = point[1]
                    p_cate = cate_points[i]
                    p_type = type_points[i]
                    point_data = p_cate + '_' + p_type + ',' + str(x) + ',' + str(y) + ',' + img_filename + ',' + str(img_w) + ',' + str(img_h)
                    print(point_data)
                    point_list.append(point_data)

                if(len(point_list)>0):
                    img_target_file = os.path.join(target_ds_path, target_images, img_filename)
                    json_target_file = os.path.join(target_ds_path, target_labels, json_filename)
                    makeLabelFile(json_target_file, point_list, img_target_file, img)
