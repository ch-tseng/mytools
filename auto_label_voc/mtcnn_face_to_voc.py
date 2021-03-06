<<<<<<< HEAD
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
from imutils.face_utils import rect_to_bb
import dlib
import imutils
import os, time
import os.path
import numpy as np

#-------------------------------------------

mediaType = "video"  # image / video / webcamera
imageFolder = "/media/sf_VMShare/autoFace_dataset/images"
videoFile = "/media/sf_VMShare/face1.mp4"
videoOutFile = "/media/sf_VMShare/out_face1.mp4"

datasetPath = "/media/sf_VMShare/practice/facial"
imgPath = "images/"
labelPath = "labels/"
imgType = "jpg"  # jpg, png

minFaceSize = (60, 60)
maxImageWidth = 1200
interval_frames = 30

landmarksDB = "shape_predictor_68_face_landmarks.dat"
dlib_detectorRatio = 2
xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#-------------------------------------------

def chkEnv():
    if not os.path.exists(landmarksDB):
        print("There is no landmark db file for this path:", landmarksDB)
        quit()

    if(mediaType=="image" and (not os.path.exists(imageFolder))):
        print("There is no folder for this path:", imageFolder)
        quit()

    if(mediaType=="video" and (not os.path.exists(videoFile))):
        print("There is no video file for this path:", videoFile)
        quit()

    if not os.path.exists(datasetPath):
        os.makedirs(datasetPath)

    if not os.path.exists(os.path.join(datasetPath,imgPath)):
        os.makedirs(os.path.join(datasetPath,imgPath))

    if not os.path.exists(os.path.join(datasetPath,labelPath)):
        os.makedirs(os.path.join(datasetPath,labelPath))

def getEyebrowShapes(landmarks):
    #right eye: 17~21
    #left eye: 22~26
    right_eyebrows = []
    left_eyebrows = []

    for id in range(17,22):
        right_eyebrows.append((landmarks.part(id).x, landmarks.part(id).y))

    for id in range(22,27):
        left_eyebrows.append((landmarks.part(id).x, landmarks.part(id).y))

    eyebrows_right_np = np.array(right_eyebrows)
    eyebrows_left_np = np.array(left_eyebrows)
    bbox_right = cv2.boundingRect(eyebrows_right_np)
    bbox_left = cv2.boundingRect(eyebrows_left_np)

    return bbox_left, bbox_right

def getEyeShapes(landmarks):
    #right eye: 36~41
    #left eye: 42~47
    right_eyes = []
    left_eyes = []

    for id in range(36,42):
        right_eyes.append((landmarks.part(id).x, landmarks.part(id).y))

    for id in range(42,48):
        left_eyes.append((landmarks.part(id).x, landmarks.part(id).y))

    eyes_right_np = np.array(right_eyes)
    eyes_left_np = np.array(left_eyes)
    bbox_right = cv2.boundingRect(eyes_right_np)
    bbox_left = cv2.boundingRect(eyes_left_np)

    return bbox_left, bbox_right

def getNoseShapes(landmarks):
    #nose: 27~35
    nose = []

    for id in range(27,36):
        nose.append((landmarks.part(id).x, landmarks.part(id).y))

    nose_np = np.array(nose)
    bbox = cv2.boundingRect(nose_np)

    return bbox

def getMouthShapes(landmarks):
    #outer: 48~59
    #inner: 60~67
    outer_mouth = []
    inner_mouth = []

    for id in range(48,60):
        outer_mouth.append((landmarks.part(id).x, landmarks.part(id).y))

    for id in range(60,68):
        inner_mouth.append((landmarks.part(id).x, landmarks.part(id).y))

    outer_mouth_np = np.array(outer_mouth)
    inner_mouth_np = np.array(inner_mouth)
    bbox_outer_mouth = cv2.boundingRect(outer_mouth_np)
    bbox_inner_mouth = cv2.boundingRect(inner_mouth_np)

    return bbox_outer_mouth, bbox_inner_mouth

def getChinShapes(landmarks):
    #nose: 0~16
    chin = []

    for id in range(0,17):
        chin.append((landmarks.part(id).x, landmarks.part(id).y))

    chin_np = np.array(chin)
    bbox = cv2.boundingRect(chin_np)

    return bbox

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

def makeLabelFile(img, bboxes):
    filename = str(time.time())
    jpgFilename = filename + "." + imgType
    xmlFilename = filename + ".xml"

    cv2.imwrite(os.path.join(datasetPath, imgPath, jpgFilename), img)

    xmlContent = generateXML(img, xmlFilename, os.path.join(datasetPath ,labelPath, xmlFilename), bboxes)
    file = open(os.path.join(datasetPath, labelPath, xmlFilename), "w")
    file.write(xmlContent)
    file.close

def labelFacial(img):
    ii = 0
    BBOX_facials = {}

    detector = dlib.get_frontal_face_detector()

    #if(img.shape[1]>maxImageWidth):
    #    img = imutils.resize(img, width=maxImageWidth)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector( gray , dlib_detectorRatio)

    if(len(rects)>0):
        BBOX_faces = []
        BBOX_leftEyebrow = []
        BBOX_rightEyebrow = []
        BBOX_leftEye = []
        BBOX_rightEye = []
        BBOX_nose = []
        BBOX_outer_mouth = []
        BBOX_inner_mouth = []
        BBOX_chin = []

        ii = 0
        for faceid, rect in enumerate(rects):
            (x, y, w, h) = rect_to_bb(rect)
            if(w>minFaceSize[0] and h>minFaceSize[1]):
                BBOX_faces.append((x,y,w,h))

                shape = predictor(gray, rect)

                bbox_leftEyebrow, bbox_rightEyebrow = getEyebrowShapes(shape)
                BBOX_leftEyebrow.append(bbox_leftEyebrow)
                BBOX_rightEyebrow.append(bbox_rightEyebrow)

                bbox_leftEye, bbox_rightEye = getEyeShapes(shape)
                BBOX_leftEye.append(bbox_leftEye)
                BBOX_rightEye.append(bbox_rightEye)

                bbox_nose = getNoseShapes(shape)
                BBOX_nose.append(bbox_nose)

                bbox_outer_mouth, bbox_inner_mouth = getMouthShapes(shape)
                BBOX_outer_mouth.append(bbox_outer_mouth)
                BBOX_inner_mouth.append(bbox_inner_mouth)

                bbox_chin = getChinShapes(shape)
                BBOX_chin.append(bbox_chin)

                ii += 1

            if(ii>0):
                BBOX_facials = { "face": BBOX_faces, "lefteyebrow": BBOX_leftEyebrow, "righteyebrow": BBOX_rightEyebrow,\
                    "lefteye":BBOX_leftEye, "righteye":BBOX_rightEye, "nose":BBOX_nose,\
                    "outter_mouth": BBOX_outer_mouth, "inner_mouth": BBOX_inner_mouth, "chin":BBOX_chin }

                makeLabelFile(img, BBOX_facials)

    return ii, BBOX_facials

#--------------------------------------------

chkEnv()

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(landmarksDB)

if(mediaType=="image"):
    for file in os.listdir(imageFolder):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            print("Processing: ", os.path.join(imageFolder, file))

            image = cv2.imread(os.path.joib(imageFolder, file))
            if(image.shape[1]>maxImageWidth):
                image = imutils.resize(image, width=maxImageWidth)

            labelFacial(image)

elif(mediaType=="video" or mediaType=="webcamera"):
    if(mediaType=="video"):
        camera = cv2.VideoCapture(videoFile)
    else:
        camera = cv2.VideoCapture(0)

    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

    if(videoOutFile != ""):
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter(videoOutFile, fourcc, 30.0, (int(width),int(height)))

    grabbed = True

    i = 0
    while grabbed:
        i += 1
        (grabbed, frame) = camera.read()
        if((grabbed is True) or (i % interval_frames == 0)):
            if(frame.shape[1]>maxImageWidth):
                frame = imutils.resize(frame, width=maxImageWidth)

            numFace, dictBBOXES = labelFacial(frame)

            if(numFace>0):
                for labelName, bbox_array in dictBBOXES.items():
                    for bbox in bbox_array:
                        cv2.rectangle( frame,(bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3]),(0,255,0),2)

            if(videoOutFile != ""):
                out.write(frame)

            cv2.imshow("FRAME", imutils.resize(frame, width=640))
            print("Frame #{}".format(i))

            cv2.waitKey(1)

        else:
            out.release()
=======
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
>>>>>>> 6b86ec7a0a67833986d049d65a75bfc0275e1312
