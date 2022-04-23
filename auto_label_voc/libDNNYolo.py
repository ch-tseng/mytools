import time
import cv2
import random
import numpy as np
import math
import torch
import json
from PIL import ImageFont, ImageDraw, Image
import random

class combinePredicts:
    def __init__(self, th_grouping=[], grouping=True, iou_score=0.85):
        self.grouping = grouping
        self.th_grouping = th_grouping
        self.iou_score = iou_score

    def giou(self, boxes):

        minx, miny, maxx, maxy = 999999, 999999, 0, 0
        area = 0
        for box in boxes:
            x,y,w,h = box[0], box[1], box[2], box[3]
            if x<minx: minx = x
            if y<miny: miny = y
            if x+w > maxx: maxx = x+w
            if y+h > maxy: maxy = y+h
            area += w*h

        tarea = (maxx-minx) * (maxy-miny)
        
        return area/tarea


    def group_boxes(self, scores, cids, bboxes, cnames):
        print('===> ', bboxes)
        print('===> ', cnames)
        totals = {}
        for i, cname in enumerate(cnames):
            cid = cids[i]
            if cname not in totals:
                totals.update( { cname:[ (scores[i], bboxes[i], cid) ] } )
            else:
                clist = totals[cname]
                clist.append( (scores[i], bboxes[i], cid ))
                totals.update( { cname: clist } )

        print('   A', totals)



        n_scores, n_cids, n_bboxes, n_cnames = [], [], [], []
        for ctype in totals:        
            
            boxes_all = []
            for infer in totals[ctype]:
                boxes_all.append(infer[1])

            if ctype in self.th_grouping and len(boxes_all)>1:
                iou_score = self.giou(boxes_all)
                print('   IOU: ',ctype, iou_score)
            else:
                iou_score = 0.0

            sx, sy, mx, my, ts, count = 999999, 999999, 0, 0, 0, 0
            for infer in totals[ctype]:                        
                score = infer[0]
                bbox = infer[1]
                cid = infer[2]
                x1, y1, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
                x2, y2 = x1+w, y1+h

                if ctype in self.th_grouping and iou_score>self.iou_score:
                    if x1<sx: sx = x1
                    if y1<sy: sy = y1
                    if x2>mx: mx = x2
                    if y2>my: my = y2

                else:
                    n_scores.append( score )
                    n_cids.append(cid)
                    n_bboxes.append( [x1,y1,w,h] )
                    n_cnames.append(ctype)

                ts += score
                count += 1

            if ctype in self.th_grouping and iou_score>self.iou_score:                
                n_scores.append( round(ts/count,2))
                n_cids.append(cid)
                n_bboxes.append( [sx,sy,mx-sx,my-sy] )
                n_cnames.append(ctype)

        print('   B', n_bboxes )
        print('   B', n_cnames )
        return n_scores, n_cids, n_bboxes, n_cnames 

    def nms_totalboxes(self, boxes, confidence=0.1, cnms=0.35):
        cbboxes, cscores, labels, classes = [], [], [], []
        for cbox in boxes:
            (bboxes, scores, labelNames, classIds) = cbox
            cbboxes += bboxes
            cscores += scores
            labels += labelNames
            classes += classIds

        indices = cv2.dnn.NMSBoxes(cbboxes, cscores, confidence, cnms)
        nms_classIds, nms_confidences, nms_boxes, nms_labelNames = [], [], [], []

        for i in indices:
            try:
                i = i[0]
            except:
                pass

            box = cbboxes[i]

            nms_confidences.append(cscores[i])
            nms_classIds.append(classes[i])
            nms_boxes.append(box)
            nms_labelNames.append(labels[i])

        
        if self.grouping is True:
            nms_confidences, nms_classIds, nms_boxes, nms_labelNames = self.group_boxes(nms_confidences, nms_classIds, nms_boxes, nms_labelNames)
            

        return nms_confidences, nms_classIds, nms_boxes, nms_labelNames

class opencvYOLO:
    def __init__(self, mtype='darknet', imgsize=(416,416), objnames="coco.names", classmap=None, colors=None,\
            weights="yolov3.weights", darknetcfg="yolov3.cfg", score=0.25, nms=0.6, gpu=False):
        self.mtype = mtype
        self.imgsize = imgsize
        self.score = score
        self.nms = nms

        self.inpWidth = self.imgsize[0]
        self.inpHeight = self.imgsize[1]
        self.classes = None
        self.classmap = classmap
        self.colors = colors
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

        if self.colors is None:
            tcolors = []
            for id, cname in enumerate(self.classes):
                tcolor = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                tcolors.append(tcolor)

            self.tcolors = tcolors

        else:
            self.tcolors = self.colors

        self.net = dnn

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
        if self.classmap is not None:
            text_w, text_h = text_size
            text_w, text_h = int(2*text_w/3), text_h+int(text_h/2)
        else:
            text_w, text_h = text_size
            text_w, text_h = int(2*text_w/3), text_h+int(text_h/2)

        rx, ry = x, y-text_h

        if text_h>70: text_h = 70
        if text_h<25: text_h = 25
        if font_scale>2.2: font_scale=2.2
        if font_scale<0.75: font_scale=0.75
        rx2, ry2 = rx+text_w, ry+text_h
        if rx<0: rx =0
        if ry<0: ry =0
        if rx2>img.shape[1]: rx2=img.shape[1]
        if ry2>img.shape[0]: ry2=img.shape[0]
        #cv2.rectangle(img, (rx,ry), (rx2, ry2), text_color, -1)
        
        img = self.printText(img, labeltxt, color=(text_color_bg[0],text_color_bg[1],text_color_bg[2],0), size=font_scale, \
                pos=(x, y), type=type) 

        return img

    def printText(self, bg, txt, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
        (b,g,r,a) = color
        (x,y) = pos

        if(type=="English"):
            cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

        else:
            ## Use simsum.ttf to write Chinese.
            fontpath = "fonts/wt009.ttf"
            font = ImageFont.truetype(fontpath, int(size*10*2))
            (width, height), (offset_x, offset_y) = font.font.getsize(txt)
            y = y - height
            if y<0: y = 0

            cv2.rectangle(bg, (x,y), (x+width, y+height), (255-b, 255-g, 255-r), -1)

            img_pil = Image.fromarray(bg)
            draw = ImageDraw.Draw(img_pil)
            #print('font.getmask(txt).getbbox()', font.getmask(txt).getbbox())
            #ascent, descent = font.getmetrics()
            
            #draw.rectangle([(pos[0], pos[1]+offset_y), (width, height)], fill=(202, 229, 134))

            draw.text((x,y),  txt, font = font, fill = (b, g, r, a))
            bg = np.array(img_pil)
            

        return bg


    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs

        try:
            return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        except:
            return [layersNames[i - 1] for i in net.getUnconnectedOutLayers()]

    def postprocess(self, frame, outs, labelWant, drawBox, bold, textsize, org_frame):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        tcolors = self.tcolors
 
        classIds = []
        labelName = []
        confidences = []
        boxes = []
        boxbold = []
        labelsize = []
        #boldcolor = []
        #textcolor = []

        for testi, out in enumerate(outs):

            for testii, detection in enumerate(out):

                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                label = self.classes[classId]
                
                if( (labelWant=="" or (label in labelWant)) and (confidence > self.score) ):
                    #print(detection)
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    if org_frame is not None:
                        center_y += org_frame.shape[0] - frame.shape[0]

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
                    #boldcolor.append(bcolor)
                    #textcolor.append(tcolor)


        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.score, self.nms)
        self.indices = indices

        nms_classIds = []
        #labelName = []
        nms_confidences = []
        nms_boxes = []
        nms_boxbold = []
        nms_labelNames = []

        if org_frame is not None:
            frame = org_frame.copy()

        for i in indices:
            try:
                i = i[0]
            except:
                pass

            box = boxes[i]
            left = box[0]
            if left<0: left = 0
            top = box[1]
            if top<0: top = 0
            width = box[2]
            if left+width>frame.shape[1]:
                width = frame.shape[1]-left

            height = box[3]
            if top+height>frame.shape[0]:
                height = frame.shape[0]-top

            nms_confidences.append(confidences[i])
            nms_classIds.append(classIds[i])
            nms_boxes.append((left,top,width,height))
            nms_labelNames.append(labelName[i])

            if(drawBox==True):
                txt_color = tcolors[classIds[i]]

                frame = self.drawPred(frame, labelName[i], confidences[i], boxbold[i], txt_color,
                    left, top, left + width, top + height)


        self.bbox = nms_boxes
        self.classIds = nms_classIds
        self.scores = nms_confidences
        self.labelNames = nms_labelNames
        self.frame = frame

        
        return frame

    def drawPred(self, frame, cid, conf, bold, textcolor, left, top, right, bottom, type='Chinese'):
        if self.classmap is not None:
            className = self.classmap[cid]
        else:
            className = cid

        label = '{}({}%)'.format(className, int(conf*100))
        #print('label', label)
        border_rect = 2
        if(frame.shape[0]<720): border_rect = 1

        textsize = (right - left) / 250.0
        txtbgColor = (255-textcolor[0], 255-textcolor[1], 255-textcolor[2])
        txtdata = (cv2.FONT_HERSHEY_SIMPLEX, textsize, border_rect, textcolor, txtbgColor)

        #print('counts', len(className))
        if left<0: left = 0
        if top<0: top = 0
        if right>frame.shape[1]: right=frame.shape[1]
        if bottom>frame.shape[0]: bottom=frame.shape[0]
        
        cv2.rectangle(frame, (left, top), (right, bottom), textcolor, border_rect)
        frame = self.bg_text(frame, label, (left+1, top+1), txtdata, type=type)

        #print(frame.shape)
        return frame

    def getObject(self, frame, score, nms, labelWant='', drawBox=False, char_type='Chinese', org_frame=None):
        textsize = 0.8
        tcolors = self.tcolors
        bold = 1
        if frame.shape[0]>720 and frame.shape[1]>1024: bold = 2

        if self.mtype == 'yolov5':
            self.net.conf = score  # confidence threshold (0-1)
            self.net.iou = nms  # NMS IoU threshold (0-1)

            results = self.net(frame[...,::-1], size=self.imgsize[0])
            predictions = json.loads(results.pandas().xyxy[0].to_json(orient="records"))

            bboxes, names, cids, scores = [], [], [], []
            for p in predictions:
                if not p['name'] in labelWant:
                    continue

                xmin, xmax, ymin, ymax = int(p['xmin']), int(p['xmax']), int(p['ymin']), int(p['ymax'])

                if org_frame is not None:
                    ymin = ymin + (org_frame.shape[0]-frame.shape[0])
                    ymax = ymax + (org_frame.shape[0]-frame.shape[0])

                bboxes.append((xmin,ymin,xmax-xmin,ymax-ymin))
                scores.append(float(p['confidence']))
                names.append(p['name'])
                cids.append(p['class'])

                if(drawBox==True):
                    txt_color = tcolors[p['class']]

                    org_frame = self.drawPred(org_frame, p['name'], float(p['confidence']), bold, txt_color,
                        xmin, ymin, xmax, ymax, type=char_type)

            if org_frame is not None:
                frame = org_frame.copy()


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

            frame = self.postprocess(frame, outs, labelWant, drawBox, bold, textsize, org_frame)
            self.objCounts = len(self.indices)
            # Put efficiency information. The function getPerfProfile returns the 
            # overall time for inference(t) and the timings for each of the layers(in layersTimes)
            #t, _ = net.getPerfProfile()

        self.frame = frame

        return frame
