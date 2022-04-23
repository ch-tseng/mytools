from libDNNYolo import opencvYOLO
import cv2
import imutils
import time, os, glob
import random
import numpy as np
import torch

target_dataset = r'G:\FINAL_SALE\face_eye_ball\dataset'
output_dataset = r'G:\FINAL_SALE\face_eye_ball\dataset_new'
labels_want = [ 'person_head', 'person_vbox' ]
custom_model = True
pretrained_model_type = 'yolov5'

#custom_model
model_size = (1280,1280)
path_objname = r"L:\ModelSale_2022\Crowded_Human\models\obj.names"
path_weights = r"L:\ModelSale_2022\Crowded_Human\models\yolov5x6\yolov5x6.pt"
path_darknetcfg = r"models\cfg.road_server_v2\yolov3.cfg"
score = 0.25
nms = 0.45
gpu = False

#pretrained model

def create_model(path_weights, model_size, path_objname, path_darknetcfg, score, nms, gpu):
    
    if custom_model is True:
        if path_weights[-2:] == 'pt':
            model = opencvYOLO( \
                mtype='yolov5', imgsize=model_size, \
                objnames=path_objname, \
                classmap=None, colors=None, \
                weights=path_weights, \
                darknetcfg='', score=score, nms=nms, gpu=gpu)

        else:
            model = opencvYOLO( \
                mtype='darknet', imgsize=model_size, \
                objnames=path_objname, \
                classmap=None, \
                weights=path_weights, \
                darknetcfg=path_darknetcfg, \
                score=score, nms=nms, gpu=gpu)

    else:
        if pretrained_model_type == 'yolov5':
            model = opencvYOLO( \
                mtype='yolov5', imgsize=(1280,1280), \
                objnames='models/pretrained/yolov4/obj.names', \
                classmap=None, colors=None, \
                weights='models/pretrained/yolov5/yolov5x6.pt', \
                darknetcfg='', score=0.35, nms=0.5, gpu=False)

        else:
            model = opencvYOLO( \
                mtype='darknet', imgsize=(608,608), \
                objnames='models/pretrained/yolov4/obj.names', \
                classmap=None, \
                weights='models/pretrained/yolov4/yolov4.weights', \
                darknetcfg='models/pretrained/yolov4/yolov4.cfg', \
                score=0.35, nms=0.5, gpu=False)

    return model

def get_img():
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

def makeLabelFile(img, bboxes, img_file_name):
    img_path = os.path.join(target_dataset, 'images')
    file_name, ext_name = os.path.splitext(img_file_name)
    jpgFilename = file_name + ext_name    
    xmlFilename = file_name + ".xml"

    org_xml_file = os.path.join(target_dataset, 'labels', xmlFilename )
    output_xml_file = os.path.join(output_dataset, 'labels', xmlFilename)    
    print('    org_xml_file', org_xml_file)
    print('    output_xml_file', output_xml_file)
    xmlObject = ""
    if os.path.exists(org_xml_file):
        for labelName, bbox_array in bboxes.items():
            for bbox in bbox_array:
                xmlObject = xmlObject + writeObjects(labelName, bbox)

        with open(org_xml_file, 'r') as f:
            org_xml_data = f.read()

        new_xml_data = org_xml_data.replace('</annotation>', xmlObject+'</annotation>')

        with open(output_xml_file, 'w') as f:
            f.write(new_xml_data)

    else:
        xmlContent = generateXML(img, xmlFilename, output_xml_file, bboxes)
        file = open(output_xml_file, "w")
        file.write(xmlContent)
        file.close    

    cv2.imwrite(os.path.join(output_dataset, 'images', jpgFilename) , img)


target_dataset = target_dataset.replace('\\', '/')
path_objname = path_objname.replace('\\', '/')
path_weights = path_weights.replace('\\', '/')
path_darknetcfg = path_darknetcfg.replace('\\', '/')

if not os.path.exists( output_dataset):
    os.makedirs(output_dataset)

if not os.path.exists( os.path.join(output_dataset, 'labels')):
    os.makedirs(os.path.join(output_dataset, 'labels'))
if not os.path.exists( os.path.join(output_dataset, 'images')):
    os.makedirs(os.path.join(output_dataset, 'images'))   

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"
obj_count = {}

img_id = 0
img_list = glob.glob(os.path.join(target_dataset, 'images','*.jpg'))
img_list += glob.glob(os.path.join(target_dataset, 'images', '*.bmp'))
img_list += glob.glob(os.path.join(target_dataset, 'images', '*.jpeg'))

yolomodel = create_model(path_weights, model_size, path_objname, path_darknetcfg, score, nms, gpu)

if __name__ == "__main__":

    print('Push Q to quit the program.')
    INPUT = img_list

    while True:
        hasFrame, frame, frame_name, (width, height) = get_img()
        img = frame.copy()

        img = yolomodel.getObject(img, score, nms, labelWant=labels_want, drawBox=True, char_type='Chinese', org_frame=frame.copy())
        cv2.imshow('test', imutils.resize(img, height=400))
        k = cv2.waitKey(1)
        if(k==113):
            break

        bbox_objects = {}
        for i, label in enumerate(yolomodel.labelNames):

            box = yolomodel.bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            if(label in obj_count):
                counts = obj_count[label]+1
            else:
                counts = 1

            obj_count.update({label:counts})

            if(label in bbox_objects):
                bbox_objects[label].append([x,y,w,h])
            else:
                bbox_objects[label] = [[x,y,w,h]]

            #print("save object file:", filename)
            #cv2.imwrite(os.path.join(folder_path, filename), img_area)

        if(len(bbox_objects)>0):
            print('get image:', frame_name)
            makeLabelFile(frame, bbox_objects, frame_name)