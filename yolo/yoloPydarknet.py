import time
import cv2
import numpy as np
from pydarknet import Detector, Image

'''
from yoloPydarknet import pydarknetYOLO
import cv2
import imutils

yolo = pydarknetYOLO( obdata="../darknet/cfg/coco.data", weights="yolov3.weights", cfg="../../darknet/cfg/yolov3.cfg")

if __name__ == "__main__":

    VIDEO_IN = cv2.VideoCapture("/media/sf_ShareFolder/IMG_8055.m4v")

    frameID = 0
    while True:
        hasFrame, frame = VIDEO_IN.read()
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            cv2.waitKey(3000)
            break

        yolo.getObject(frame, labelwant="", drawBox=True)
        print ("Object counts:", yolo.objCounts)
        yolo.listLabels()
        print("ID #1:", yolo.list_Label(1))
        cv2.imshow("Frame", imutils.resize(frame, width=850) )
        cv2.waitKey(1)

'''
class pydarknetYOLO():
    def __init__(self, obdata="coco.data", weights="yolov3.weights", cfg="yolov3.cfg"):
        self.score = 0.5

        net = Detector(bytes(cfg, encoding="utf-8"), 
               bytes(weights, encoding="utf-8"), 0,
               bytes(obdata, encoding="utf-8"))

        self.net = net

    def setScore(self, score=0.5):
        self.score = score

    def getObject(self, frame, labelWant=("car","person"), drawBox=False, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255)):
        net = self.net
        dark_frame = Image(frame)
        self.results = net.detect(dark_frame)

        self.objCounts = len(self.results)

        boxes = []
        scores = []
        labels = []
        boxbold = []
        labelsize = []
        boldcolor = []
        textcolor = []

        for cat, score, box in self.results:
            label = cat.decode('utf-8')

            left = int(box[0] - (box[2]/2))
            top = int(box[1] - (box[3]/2))
            width = int(box[2])
            height = int(box[3])

            print("Label:{}, score:{}, left:{}, top:{}, right:{}, bottom:{}".format(label, 
                score, left, top, left + width, top + height) )

            if(labelWant=="" or (label in labelWant)):
                boxes.append( (left, top, width, height) )
                scores.append(score)
                labels.append(label) 
                boxbold.append(bold)
                labelsize.append(textsize)
                boldcolor.append(bcolor)
                textcolor.append(tcolor)

                if(drawBox==True):
                    self.drawPred(frame, label, score,  bold, bcolor, tcolor,
                    textsize, left, top, left+width, top+height)

        self.bbox = boxes
        self.scores = scores
        self.labels = labels

    def listLabels(self):
        for i, box in enumerate(self.bbox):
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]

            label = self.labels[i]
            score = self.scores[i]
            print("Label:{}, score:{}, left:{}, top:{}, right:{}, bottom:{}".format(label, score, left, top, left + width, top + height) )

    def list_Label(self, id):
        box = self.bbox[id]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        label = self.labels[id]
        score = self.scores[id]

        return (left, top, width, height, label, score)

    # Draw the predicted bounding box
    def drawPred(self, frame, label, conf, bold, boldcolor, textcolor, textsize, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), boldcolor, bold)

        score = '%.2f' % conf

        label = '%s:%s' % (label, score)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, textsize, textcolor)

