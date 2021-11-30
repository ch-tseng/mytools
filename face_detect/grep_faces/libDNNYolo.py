import time
import cv2
import random
import numpy as np
import math
import torch
import json
from PIL import ImageFont, ImageDraw, Image
import random

class opencvYOLO:
    def __init__(self, mtype='darknet', imgsize=(416,416), objnames="coco.names", \
            weights="yolov3.weights", darknetcfg="yolov3.cfg", score=0.25, nms=0.6, tcolors=None, gpu=False):
        self.mtype = mtype
        self.imgsize = imgsize
        self.score = score
        self.nms = nms

        self.inpWidth = self.imgsize[0]
        self.inpHeight = self.imgsize[1]
        self.classes = None
        with open(objnames, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        if mtype == 'yolov5':
            dnn = torch.hub.load('ultralytics/yolov5', 'custom', weights, force_reload=True)
            dnn.conf = score
            dnn.iou = nms
        else:
            dnn = cv2.dnn.readNetFromDarknet(darknetcfg, weights)

            if gpu is True:
                dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
            else:
                dnn.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                dnn.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)


        self.net = dnn

        if tcolors is None:
            tcolors = []
            for id, cname in enumerate(self.classes):
                tcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                tcolors.append(tcolor)

        self.tcolors = tcolors
        print('tcolors', self.tcolors)

    def setScore(self, score=0.5):
        if self.mtype == 'yolov5':
            self.net.conf = score
        else:
            self.score = score

    def setNMS(self, nms=0.8):
        if self.mtype == 'yolov5':
            self.net.iou = nms
        else:
            self.nms = nms

    def bg_text(self, img, labeltxt, loc, txtdata, type="Chinese"):
        (x,y) = loc
        (font, font_scale, font_thickness, text_color, text_color_bg) = txtdata

        #max_scale =(img.shape[1]/1920) * 2
        #if font_scale>max_scale: font_scale = max_scale
        text_size, _ = cv2.getTextSize(labeltxt, font, font_scale, font_thickness)
        text_w, text_h = text_size
        text_w, text_h = int(2*text_w/3), text_h+int(text_h/2)
        rx, ry = x, y-2

        if text_h>120: text_h = 120
        if font_scale>4: font_scale=4
        rx2, ry2 = rx+text_w, ry+text_h
        if rx<0: rx =0
        if ry<0: ry =0
        if rx2>img.shape[1]: rx2=img.shape[1]
        if ry2>img.shape[0]: ry2=img.shape[0]
        cv2.rectangle(img, (rx,ry), (rx2, ry2), text_color_bg, -1)

        img = self.printText(img, labeltxt, color=(text_color[0],text_color[1],text_color[2],0), size=font_scale, \
                pos=(x, y), type=type) 

        return img

    def printText(self, bg, txt, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
        (b,g,r,a) = color

        if(type=="English"):
            cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

        else:
            ## Use simsum.ttf to write Chinese.
            fontpath = "fonts/wt009.ttf"
            font = ImageFont.truetype(fontpath, int(size*10*2))
            img_pil = Image.fromarray(bg)
            draw = ImageDraw.Draw(img_pil)
            draw.text(pos,  txt, font = font, fill = (b, g, r, a))
            bg = np.array(img_pil)

        return bg

    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    def postprocess(self, frame, outs, labelWant, drawBox, bold, textsize):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        tcolors = self.tcolors
 
        classIds = []
        labelName = []
        confidences = []
        boxes = []
        boxbold = []
        labelsize = []
        reference_final = {}

        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                label = self.classes[classId]
                if( (labelWant=="" or (label in labelWant)) and (confidence > self.score) ):

                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append((left, top, width, height))
                    boxbold.append(bold)
                    labelName.append(label)
                    labelsize.append(textsize)

                    if classId in reference_final:
                        box_list = reference_final[classId][0]
                        score_list = reference_final[classId][1]
                    else:
                        box_list = []
                        score_list = []

                    box_list.append((left, top, width, height))
                    score_list.append(float(confidence))
                    reference_final.update( {classId:[box_list, score_list] } )


        nms_classIds = []
        #labelName = []
        nms_confidences = []
        nms_boxes = []
        nms_boxbold = []
        nms_labelNames = []

        indices = []
        for r_class_id in reference_final:
            r_boxes = reference_final[r_class_id][0]
            r_scores = reference_final[r_class_id][1]
            nms_indices = cv2.dnn.NMSBoxes(r_boxes, r_scores, self.score, self.nms)


            for ii in nms_indices:
                try:
                    i = ii[0]
                except:
                    i = ii

                box = r_boxes[i]
                left = r_boxes[i][0]
                top = r_boxes[i][1]
                width = r_boxes[i][2]
                height = r_boxes[i][3]

                nms_confidences.append(r_scores[i])
                nms_classIds.append(r_class_id)
                nms_boxes.append(box)
                nms_labelNames.append(self.classes[r_class_id])

                if(drawBox==True):
                    txt_color = tcolors[r_class_id]

                    frame = self.drawPred(frame, r_class_id, r_scores[i], 2, txt_color,
                        left, top, left + width, top + height)

        self.bbox = nms_boxes
        self.classIds = nms_classIds
        self.scores = nms_confidences
        self.labelNames = nms_labelNames
        self.frame = frame

        return frame

    def drawPred(self, frame, className, conf, bold, textcolor, left, top, right, bottom, type='Chinese'):
        if self.mtype == 'darknet':
            className = self.classes[int(className)]


        label = '{}({}%)'.format(className, int(conf*100))
        border_rect = 2
        if(frame.shape[0]<720): border_rect = 1

        textsize = (right - left) / 250.0
        txtbgColor = (255-textcolor[0], 255-textcolor[1], 255-textcolor[2])
        txtdata = (cv2.FONT_HERSHEY_SIMPLEX, textsize, border_rect, textcolor, txtbgColor)

        if left>0 and top>0 and right<frame.shape[1] and bottom<frame.shape[0]:
            cv2.rectangle(frame, (left, top), (right, bottom), textcolor, border_rect)
            #frame = self.bg_text(frame, label, (left+1, top+1), txtdata, type=type)
        return frame

    def getObject(self, frame, score, nms, labelWant='', drawBox=False, char_type='Chinese'):
        textsize = 0.8
        tcolors = self.tcolors
        bold = 1
        if frame.shape[0]>720 and frame.shape[1]>1024: bold = 2

        if self.mtype == 'yolov5':
            self.net.conf = score  # confidence threshold (0-1)
            self.net.iou = nms  # NMS IoU threshold (0-1)

            results = self.net(frame[...,::-1], size=self.imgsize[0])
            print(results.pandas())
            predictions = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

            bboxes, names, cids, scores = [], [], [], []
            for p in predictions:
                xmin, xmax, ymin, ymax = int(p['xmin']), int(p['xmax']), int(p['ymin']), int(p['ymax'])
                bboxes.append((xmin,ymin,xmax-xmin,ymax-ymin))
                scores.append(float(p['confidence']))
                names.append(p['name'])
                cids.append(p['class'])

                if(drawBox==True):
                    txt_color = tcolors[p['class']]

                    frame = self.drawPred(frame, p['name'], float(p['confidence']), bold, txt_color,
                        xmin, ymin, xmax, ymax, type=char_type)

            self.bbox = bboxes
            self.classIds = cids
            self.scores = scores
            self.labelNames = names


        else:
            net = self.net
            blob = cv2.dnn.blobFromImage(frame, 1/255, (self.inpWidth, self.inpHeight), [0,0,0], 1, crop=False)
            net.setInput(blob)
            # Runs the forward pass to get output of the output layers
            outs = net.forward(self.getOutputsNames(net))
            # Remove the bounding boxes with low confidence
            frame = self.postprocess(frame, outs, labelWant, drawBox, bold, textsize)



        self.frame = frame

        return frame
