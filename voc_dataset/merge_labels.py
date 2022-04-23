import cv2
import imutils
import time, os, glob
import random
import numpy as np
from xml.dom import minidom

#ds_images_path include 'labels' and 'images'
ds_path = r'D:\temp\dataset_new\org_dataset'
user_works = {
                'Linda': ( ['0','nose'], r'D:\temp\dataset_new\linda\labels'),
                'John':( ['person_head','face'], r'D:\temp\dataset_new\John\labels'),
                'Mary':( ['mouth'], r'D:\temp\dataset_new\John\labels')
            }

output_ds_path = r'D:\temp\dataset_new\combined_dataset'


def get_img(INPUT):
    global img_id

    frame = None
    frame_ok = False
    frame_name = None
    w,h = None, None
    if img_id<len(INPUT):
        while frame_ok is False and img_id<len(INPUT):

            file_path = INPUT[img_id]
            frame = cv2.imread(file_path)
            img_id += 1
            frame_ok = True
            try:
                (h,w,_) = frame.shape               

            except:
                print('error image, cannot read:', INPUT[img_id])                
                continue

        frame_name = os.path.basename(file_path)
        

    return frame_ok, frame, frame_name, (w,h)

def getLabels(xmlFile, labelwant=[]):
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

    labels, bboxes = [], []
    for id, lname in enumerate(labelName):
        if lname in labelwant:
            labels.append(lname)
            bboxes.append((labelXmin[id], labelYmin[id], labelXmax[id], labelYmax[id]))

    return labels, bboxes

def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[3]))

    return file_updated


ds_path = ds_path.replace('\\', '/')
output_ds_path = output_ds_path.replace('\\', '/')
output_xml_path = os.path.join(output_ds_path, 'labels')
output_img_path = os.path.join(output_ds_path, 'images')
if not os.path.exists(output_xml_path):
    os.makedirs(output_xml_path)
if not os.path.exists(output_img_path):
    os.makedirs(output_img_path)    

xml_samplefile = "xml_file.txt"
object_xml_file = "xml_object.txt"
obj_count = {}

img_id = 0
img_list = glob.glob(os.path.join(ds_path, 'images','*.jpg'))
img_list += glob.glob(os.path.join(ds_path, 'images', '*.bmp'))
img_list += glob.glob(os.path.join(ds_path, 'images', '*.jpeg'))

if __name__ == "__main__":

    print('Push Q to quit the program.')
    while True:
        hasFrame, frame, frame_name, (width, height) = get_img(img_list)
        img = frame.copy()
        file_name, ext_name = os.path.splitext(frame_name) 

        ds_img_path = os.path.join(ds_path, 'images', frame_name)
        ds_xml_path = os.path.join(ds_path, 'labels', file_name+'.xml')     

        xmlObject = ""
        for username in user_works:
            user_labels = user_works[username][0]
            user_xml_path = os.path.join(user_works[username][1], file_name+'.xml')

            labelNames, bboxes = getLabels(user_xml_path, user_labels)
            for id in range(0, len(labelNames)):
                xmlObject += writeObjects(labelNames[id], bboxes[id])

        org_xml_data = ''
        if os.path.exists(ds_xml_path):
            with open(ds_xml_path, 'r') as f:
                org_xml_data = f.read()

        if org_xml_data == '':
            with open(xml_samplefile) as file:
                xmlfile = file.read()

            (h, w, ch) = img.shape
            xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
            xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
            xmlfile = xmlfile.replace( "{FILENAME}", frame_name )
            xmlfile = xmlfile.replace( "{PATH}", ds_img_path )
            xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

        else:
            xmlfile = org_xml_data.replace('</annotation>', xmlObject+'</annotation>')

        output_xml_file = os.path.join(output_xml_path, file_name+'.xml' )

        with open(output_xml_file, 'w') as f:
            f.write(xmlfile)

        cv2.imwrite( os.path.join(output_img_path, frame_name), frame)

        cv2.imshow('test', imutils.resize(img, height=400))
        k = cv2.waitKey(1)
        if(k==113):
            break
