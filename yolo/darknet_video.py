from ctypes import *
import math
import random
import os
import cv2
import numpy as np
import time
import darknet
import imutils

detect_score = 0.15
#YOLO
#classes = ["bicycle", "bus", "car", "motorbike", "truck" ]
classes = ["person_head", "person_vbox" ]

#yolov3
#configPath = "/DATA1/Datasets_mine/labeled/vehicles_coco_voc/yolov3_config/yolov3.cfg"
#weightPath = "/DATA1/Datasets_mine/labeled/vehicles_coco_voc/yolov3_config/weights/yolov3_29000.weights"
#metaPath = "/DATA1/Datasets_mine/labeled/vehicles_coco_voc/yolov3_config/obj.data"
#yolov4
configPath = "/DATA1/Datasets_mine/labeled/crowndHuman_2_classes/yolov3-tiny_config/yolov3-tiny.cfg"
#weightPath = "/DATA1/Datasets_mine/labeled/crowndHuman_2_classes/yolov4_config/weights/yolov4_18000.weights"
weightPath = "/DATA1/Datasets_mine/labeled/crowndHuman_2_classes/yolov3-tiny_config/weights/yolov3-tiny_1480093.weights"
metaPath = "/DATA1/Datasets_mine/labeled/crowndHuman_2_classes/yolov3-tiny_config/obj.data"

media = "/DATA1/Videos/EarthCam Live Times Square Crossroads Cam.mp4"
rotate_video = 0
write_output = True
output_video_path = "/DATA1/Outputs/Darknet_v4_Shibuya Crossing Full HD 渋谷駅 東京.avi"

#video_size = (1920, 1080)  #x,y
video_rate = 24.0

#colors = [np.random.randint(0, 256, 3).tolist() for _ in range(num_classes)]
#print("Colors:", colors)
colors = [[10, 211, 152], [32, 146, 218], [193, 57, 12], [1, 231, 13], [46, 154, 202]]

def display_counts(img, list_txt, loc, font_size, font_color, font_bold):

    y_move = 0
    for txt in list_txt:
        cv2.putText(img, txt, (loc[0], loc[1]+y_move), cv2.FONT_HERSHEY_SIMPLEX, font_size, font_color, font_bold)
        y_move += 35

    return img

def convertBack(x, y, w, h):
    xmin = int(round(x - (w / 2)))
    xmax = int(round(x + (w / 2)))
    ymin = int(round(y - (h / 2)))
    ymax = int(round(y + (h / 2)))
    return xmin, ymin, xmax, ymax

def draw_boxes(image, boxes, scores, labels, colors, classes, font_size, font_border, box_border, print_labels):
    for b, class_name, s in zip(boxes, labels, scores):

        x, y, w, h = list(map(int, b))
        #xmin, ymin, xmax, ymax = list(map(int, b))
        #xmax = xmin + w
        #ymax = ymin + h
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))

        score = '{:.4f}'.format(s)
        color = colors[classes.index(class_name)]
        label = '-'.join([class_name, score])

        ret, baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_size, font_border)
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, box_border)
        cv2.rectangle(image, (xmin, ymax - ret[1] - baseline), (xmin + ret[0], ymax), color, -1)
        if(print_labels is True):
            cv2.putText(image, label, (xmin, ymax - baseline), cv2.FONT_HERSHEY_SIMPLEX, font_size, (0, 0, 0), font_border)

    return image

def cvDrawBoxes(detections, img, w_r, h_r):
    for detection in detections:
        x, y, w, h = detection[2][0] * w_r,\
            detection[2][1] * h_r,\
            detection[2][2] * w_r,\
            detection[2][3] * h_r
        xmin, ymin, xmax, ymax = convertBack(
            float(x), float(y), float(w), float(h))
        pt1 = (xmin, ymin)
        pt2 = (xmax, ymax)
        cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(img,
                    detection[0].decode() +
                    " [" + str(round(detection[1] * 100, 2)) + "]",
                    (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    [0, 255, 0], 2)
    return img


netMain = None
metaMain = None
altNames = None

def YOLO():

    global metaMain, netMain, altNames
    #configPath = configPath
    #weightPath = weightPath
    #metaPath = metaPath
    if not os.path.exists(configPath):
        raise ValueError("Invalid config path `" +
                         os.path.abspath(configPath)+"`")
    if not os.path.exists(weightPath):
        raise ValueError("Invalid weight path `" +
                         os.path.abspath(weightPath)+"`")
    if not os.path.exists(metaPath):
        raise ValueError("Invalid data file path `" +
                         os.path.abspath(metaPath)+"`")
    if netMain is None:
        netMain = darknet.load_net_custom(configPath.encode(
            "ascii"), weightPath.encode("ascii"), 0, 1)  # batch size = 1
    if metaMain is None:
        metaMain = darknet.load_meta(metaPath.encode("ascii"))
    if altNames is None:
        try:
            with open(metaPath) as metaFH:
                metaContents = metaFH.read()
                import re
                match = re.search("names *= *(.*)$", metaContents,
                                  re.IGNORECASE | re.MULTILINE)
                if match:
                    result = match.group(1)
                else:
                    result = None
                try:
                    if os.path.exists(result):
                        with open(result) as namesFH:
                            namesList = namesFH.read().strip().split("\n")
                            altNames = [x.strip() for x in namesList]
                except TypeError:
                    pass
        except Exception:
            pass
    #cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(media)
    cap.set(3, 1280)
    cap.set(4, 720)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    print("width, height: ", width, height)

    if(write_output is True):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        #out = cv2.VideoWriter(
        #    "output.avi", cv2.VideoWriter_fourcc(*"MJPG"), output_video_rate,
        #    (darknet.network_width(netMain), darknet.network_height(netMain)))
        out = cv2.VideoWriter(output_video_path, fourcc, video_rate, (int(width),int(height)))

    print("Starting the YOLO loop...")

    # Create an image we reuse for each detect
    darknet_image = darknet.make_image(darknet.network_width(netMain),
                                    darknet.network_height(netMain),3)
    start_time = time.time()
    frame_id = 0
    while True:
        prev_time = time.time()
        ret, frame_read = cap.read()
        if ret is not True:
            break

        if(rotate_video>0):
            frame_read = imutils.rotate(frame_read, rotate_video)

        frame_rgb = cv2.cvtColor(frame_read, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb,
                                   (darknet.network_width(netMain),
                                    darknet.network_height(netMain)),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image,frame_resized.tobytes())

        detections = darknet.detect_image(netMain, metaMain, darknet_image, thresh=detect_score)
        print(detections)

        w_ratio = frame_read.shape[1] / frame_resized.shape[1]
        h_ratio = frame_read.shape[0] / frame_resized.shape[0]

        #image = cvDrawBoxes(detections, frame_read, w_ratio, h_ratio)

        boxes, scores, labels = [], [], []
        for detection in detections:
            x, y, w, h = detection[2][0] * w_ratio,\
                detection[2][1] * h_ratio,\
                detection[2][2] * w_ratio,\
                detection[2][3] * h_ratio

            boxes.append([x, y, w, h])
            labels.append(detection[0].decode())
            scores.append(round(detection[1] * 100, 2))

        classes_count = {}
        for class_name in classes:
            classes_count.update( { class_name:0 } )

        for label in labels:
            num = classes_count[label]
            classes_count.update( {label:num+1 } )

        txt_lines = []
        for txt in classes_count:
            line = "{}: {}".format(txt, classes_count[txt])
            txt_lines.append(line)


        #print("Drawbox:", boxes, scores, labels)
        image = draw_boxes(frame_read, boxes, scores, labels, colors, classes, 0.65, 1, 1, False)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(1/(time.time()-prev_time))

        image = display_counts(image, txt_lines, (image.shape[1]-350, 50), 1.25, (0,0,0), 2)
        frame_id += 1
        #cv2.imshow('Demo', image)
        if(write_output is True):
            out.write(image)

        #cv2.waitKey(1)
    cap.release()

    if(write_output is True):
        out.release()

    total_time = time.time() - start_time
    print(output_video_path, frame_id, total_time, "FPS:", round(frame_id/total_time, 5))

if __name__ == "__main__":
    YOLO()
