# -*- coding: utf-8 -*-

import cv2
import os
import imutils
from mtcnn.mtcnn import MTCNN

template_base = "landmarks_template_labeme.json"
template_shapes_a = "shape_lm_content_template.txt"
template_shapes_b = "shape_face_content_template.txt"
peoples_folder = "test_data\\"

extract_folder = "faces\\"

detector = MTCNN()

with open(template_shapes_a) as file:
    jsonfile_a = file.read()
    
with open(template_shapes_b) as file:
    jsonfile_b = file.read()    
    
with open(template_base) as file:
    basefile = file.read()    

if not os.path.exists(extract_folder):
    os.makedirs(extract_folder)


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
    
def make_marks_jsonfile(json_shapes, img_shape, file_name, img_folder):
    json_labelfile = basefile.replace("{SHAPES}", json_shapes)
    json_labelfile = json_labelfile.replace("{img_path}", file_name)
    json_labelfile = json_labelfile.replace("{img_h}", str(img_shape[0]))
    json_labelfile = json_labelfile.replace("{img_w}", str(img_shape[1]))
    base_filename, ext_filename = os.path.splitext(file_name)
    file = open(os.path.join(img_folder, base_filename+".json"), "w")
    file.write(json_labelfile)
    file.close
    
def land_marks(lm, fb_bbox, json_content, ltype=0):
    nose_x = lm[0][0]
    nose_y = lm[0][1]
    rmouth_x = lm[1][0]
    rmouth_y = lm[1][1]
    lmouth_x = lm[2][0]
    lmouth_y = lm[2][1]
    reye_x = lm[3][0]
    reye_y = lm[3][1]
    leye_x = lm[4][0]
    leye_y = lm[4][1]

    if(ltype==0):
        nose_x = nose_x - fb_bbox[0]
        nose_y = nose_y - fb_bbox[1]
        rmouth_x = rmouth_x - fb_bbox[0]
        rmouth_y = rmouth_y - fb_bbox[1]
        lmouth_x = lmouth_x - fb_bbox[0]
        lmouth_y = lmouth_y - fb_bbox[1]
        reye_x = reye_x - fb_bbox[0]
        reye_y = reye_y - fb_bbox[1]
        leye_x = leye_x - fb_bbox[0]
        leye_y = leye_y - fb_bbox[1]

    
    file_updated = json_content.replace("{nose}", "nose")
    file_updated = file_updated.replace("{x_nose}", str(nose_x))
    file_updated = file_updated.replace("{y_nose}", str(nose_y))
    file_updated = file_updated.replace("{mouth_right}", "mouth_right")

    file_updated = file_updated.replace("{x_mouth_right}", str(rmouth_x))
    file_updated = file_updated.replace("{y_mouth_right}", str(rmouth_y))
    file_updated = file_updated.replace("{mouth_left}", "mouth_left")
    file_updated = file_updated.replace("{x_mouth_left}", str(lmouth_x))
    file_updated = file_updated.replace("{y_mouth_left}", str(lmouth_y))
    file_updated = file_updated.replace("{right_eye}", "right_eye")
    file_updated = file_updated.replace("{x_right_eye}", str(reye_x))
    file_updated = file_updated.replace("{y_right_eye}", str(reye_y))
    file_updated = file_updated.replace("{left_eye}", "left_eye")
    file_updated = file_updated.replace("{x_left_eye}", str(leye_x))
    file_updated = file_updated.replace("{y_left_eye}", str(leye_y))

    return file_updated
        
for file in os.listdir(peoples_folder):
    
    
    base_filename, ext_filename = os.path.splitext(file)
  
    
    if(ext_filename.lower() in (".jpg", ".jpeg", ".bmp", ".png")):
        file_path = os.path.join(peoples_folder, file)
        
        try:        
            pic = cv2.imread(file_path)
            print(pic.shape, file_path)
        except:
            continue
        
        score, faces, landmarks = getFaces(pic)
        print("Num:", len(faces))
        print(score, faces, landmarks)
        #file_updated = ""
        json_shapes_a, json_shapes_b = "", ""
        for i, (x,y,w,h) in enumerate(faces):
            
            pic_display = pic.copy()
            cv2.rectangle( pic_display,(x,y),(x+w,y+h),(0,255,0),2)
            
            face_area = pic[y:y+h, x:x+w]
            face_filename = base_filename 
            print(face_filename + "_" + str(i) + ext_filename)
            try:
                cv2.imwrite(os.path.join(extract_folder, face_filename + "_" + str(i) + ext_filename), face_area)
            except:
                print("    ---> Error, cannot write....")
                continue
            
            #write JSON file           
            
            #Landmarks
            file_updated_a = land_marks(landmarks[i], (x,y,w,h), jsonfile_a, 0)
            if(i>0): json_shapes_a = json_shapes_a + "," + "\n"
            
            json_shapes_a = json_shapes_a + "\n" + file_updated_a
            
            file_updated_b = land_marks(landmarks[i], (x,y,w,h), jsonfile_b, 1)          
            #Face bbox
            file_updated_b = file_updated_b.replace("{face}", "face")
            file_updated_b = file_updated_b.replace("{face_lefttop_point_x}", str(x))
            file_updated_b = file_updated_b.replace("{face_lefttop_point_y}", str(y))
            file_updated_b = file_updated_b.replace("{face_righttop_point_x}", str(x+w))
            file_updated_b = file_updated_b.replace("{face_righttop_point_y}", str(y+h))
            
            make_marks_jsonfile(json_shapes_a, face_area.shape, face_filename + "_" + str(i) + ext_filename, extract_folder)
             
            if(i>0): json_shapes_b = json_shapes_b + "," + "\n"
            
            json_shapes_b = json_shapes_b + "\n" + file_updated_b

            cv2.imshow("Original", imutils.resize(pic_display, height=350))
            cv2.imshow("Face", face_area)
            cv2.waitKey(1)
            
        if(json_shapes_a != ""):
            
            make_marks_jsonfile(json_shapes_b, pic.shape, face_filename + ext_filename, peoples_folder)
            