import torch
import json
#from PIL import Image
import cv2
import imutils

media = 'swim.mp4'

#bbox
display_box = True
border = 2
bcolor = (0,255,0)

#class name
display_cname = True
display_score = True
font_size = 0.90
font_bolder = 2
fcolor = (0,255,255)

#-----------------------------------------------------------------

def detect_obj(img):
    results = model(img[...,::-1], size=640)
    predictions = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    bboxes, names, cids, scores = [], [], [], []
    for p in predictions:
        xmin, xmax, ymin, ymax = int(p['xmin']), int(p['xmax']), int(p['ymin']), int(p['ymax'])
        bboxes.append((xmin,ymin,xmax-xmin,ymax-ymin))
        scores.append(float(p['confidence']))
        names.append(p['name'])
        cids.append(p['class'])

    return (cids, names, bboxes, scores)

def draw_info(img, data):
    for name, box, score in data:
        if display_box:
            cv2.rectangle(img, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), bcolor, border)

        display_txt = ''
        if display_cname:
            display_txt += name
        if display_score:
            display_txt += str(round(score,2))
        if display_txt != '':
            cv2.putText(img, display_txt, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, font_size, fcolor, font_bolder)

    return img

# Model
model = torch.hub.load('ultralytics/yolov5', 'custom', 'weights/best.pt', force_reload=False)

model.conf = 0.25  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1)
model.classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for head, body....

camera = cv2.VideoCapture(media)
(grabbed, frame) = camera.read()

while grabbed:

    (class_ids, names, boxes, scores) = detect_obj(frame)
    frame = draw_info(frame, zip(names, boxes, scores))

    cv2.imshow('test', imutils.resize(frame, height=1024))
    cv2.waitKey(1)

    (grabbed, frame) = camera.read()
