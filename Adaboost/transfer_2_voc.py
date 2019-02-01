import cv2
import imutils
import os, time
import os.path
import numpy as np
from xml.dom import minidom

# adaboost dataset for Adaboos, folder name is label name, label files and image files in the same folder
ada_path = "/media/sf_VMshare/sunplusit_ds/hand_gesture/"
ada_label_file_ext = ".xy"
output_voc_path = "/media/sf_VMshare/sunplusit_ds/voc_hand_gesture/"
folderCharacter = "/"  # \\ is for windows
xml_samplefile = "xml_file.txt"
object_xml_file = "xml_object.txt"

#-----------------------------------------------------------------------

def chkEnv():
    if not os.path.exists(ada_path):
        print("There is no such path for Adaboost dataset:", ada_path)
        quit()

    if not os.path.exists(xml_samplefile):
        print("There is no ", xml_samplefile)
        quit()

    if not os.path.exists(object_xml_file):
        print("There is no ", object_xml_file)
        quit()

    if not os.path.exists(output_voc_path):
        print("There is no ", output_voc_path)
        quit()
    else:
        if not os.path.exists(output_voc_path + "images"):
            os.makedirs(output_voc_path + "images")
        if not os.path.exists(output_voc_path + "labels"):
            os.makedirs(output_voc_path + "labels")



def get_adaboost_bbox(adaboost_file):
    with open(adaboost_file) as file:
        file_content = file.read()
        file_content = file_content.replace("\n", "")

    box = []
    values = file_content.split(" ")
    for value in values:
        box.append(value)

    file.close

    if(len(box)>=4):
        return (int(values[0]), int(values[1]), int(values[2]), int(values[3]))
    else:
        return (None, None, None, None)

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
    xmlObject = xmlObject + writeObjects(labelName, (labelXmin, labelYmin, labelXmax, labelYmax))

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

    cv2.imwrite(output_voc_path + "images/" + jpgFilename, img)
    print("write to -->", output_voc_path + "images/" + jpgFilename)

    xmlContent = generateXML(img, xmlFilename, output_voc_path + "labels/" + xmlFilename, bboxes)
    file = open(output_voc_path + "labels/" + xmlFilename, "w")
    file.write(xmlContent)
    file.close
    print("write to -->", output_voc_path + "labels/" + xmlFilename)


#-----------------------------------------------------

chkEnv()

for className in os.listdir(ada_path):
    print("class name:", className)
    folder_path = ada_path + folderCharacter + className

    for file in os.listdir(folder_path):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            img_file = file
            label_file = filename + ada_label_file_ext

            img_path = folder_path + folderCharacter + img_file
            label_path = folder_path + folderCharacter + label_file

            if(os.path.exists(label_path)):
                (x, y, w, h) = get_adaboost_bbox(label_path)

                if(x is not None):
                    image = cv2.imread(img_path)
                    (img_width, img_height) = (image.shape[1], image.shape[0])
                    makeDatasetFile(image, img_file, (className, x, y, x+w, y+h))
