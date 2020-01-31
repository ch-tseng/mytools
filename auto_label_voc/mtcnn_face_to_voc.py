# -*- coding: utf-8 -*-

import cv2
import os
import imutils
from mtcnn.mtcnn import MTCNN

minFaceSize = (60, 60)
resize_max_width_size = 1024
label_name = "face"
peoples_folder = "H:\\working\\cascade_indoor_face\\dataset\\posSources"
target_face_voc_dataset = "H:\\working\\cascade_indoor_face\\voc_dataset"
voc_imgPath = "images/"
voc_labelPath = "labels/"
target_img_type = ".jpg"

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

detector = MTCNN()

def getFaces(img):
    faces = detector.detect_faces(img)
    scores, bboxes, landmarks = [], [], []
    for face in faces:
        x = face["box"][0]
        y = face["box"][1]
        w = face["box"][2]
        h = face["box"][3]
        
        mark0 = face["keypoints"]["nose"]
        mark1 = face["keypoints"]["mouth_right"]
        mark2 = face["keypoints"]["mouth_left"]
        mark3 = face["keypoints"]["right_eye"]
        mark4 = face["keypoints"]["left_eye"]
        
        scores.append(face["confidence"])
        
        bboxes.append((x,y,w,h))
        landmarks.append( [mark0, mark1, mark2, mark3, mark4] )

    return scores, bboxes, landmarks

def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[0] + bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[1] + bbox[3]))

    return file_updated

def generateXML(img, filename, fullpath, bboxes):
    xmlObject = ""

    for labelName, bbox_array in bboxes.items():
        for bbox in bbox_array:
            xmlObject = xmlObject + writeObjects(labelName, bbox)

    with open(xml_file) as file:
        xmlfile = file.read()

    (h, w, ch) = img.shape
    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", filename )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + filename )
    xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

    return xmlfile

def makeLabelFile(img, bboxes, filename):

    jpgFilename = filename + target_img_type
    xmlFilename = filename + ".xml"

    cv2.imwrite(os.path.join(target_face_voc_dataset, voc_imgPath, jpgFilename), img)

    xmlContent = generateXML(img, xmlFilename, os.path.join(target_face_voc_dataset ,voc_labelPath, xmlFilename), bboxes)
    file = open(os.path.join(target_face_voc_dataset, voc_labelPath, xmlFilename), "w")
    file.write(xmlContent)
    file.close    

if(not os.path.exists(target_face_voc_dataset)):
    os.mkdir(target_face_voc_dataset)
if(not os.path.exists(os.path.join(target_face_voc_dataset,voc_imgPath))):
    os.mkdir(os.path.join(target_face_voc_dataset,voc_imgPath))
if(not os.path.exists(os.path.join(target_face_voc_dataset,voc_labelPath))):
    os.mkdir(os.path.join(target_face_voc_dataset,voc_labelPath))    
        
for file in os.listdir(peoples_folder):
    
    base_filename, ext_filename = os.path.splitext(file)
    
    if(ext_filename.lower() in (".jpg", ".jpeg", ".bmp", ".png")):
        file_path = os.path.join(peoples_folder, file)
        print("file_path:", file_path)
        try:        
            pic = cv2.imread(file_path)
        except:
            print(file_path, "read error.")
            continue
        
        #cv2.imshow("TEST", pic)
        score, faces, landmarks = getFaces(pic)
        print("Faces:", faces)

        face_boxes = []
        bbox_objects = {label_name:[]}
        for i, box in enumerate(faces):
            face_boxes = bbox_objects[label_name]

            x, y, w, h = box[0], box[1], box[2], box[3]  
            face_boxes.append([x,y,w,h])          
            #filename = label + '_' + str(i) + '.jpg'
            #print("bbox_objects:", bbox_objects)

        
        if(len(face_boxes)>0):
            bbox_objects.update({label_name:face_boxes})
            makeLabelFile(pic, bbox_objects, base_filename)