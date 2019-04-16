import cv2
import imutils
import time

videoFile = "/media/sf_VMShare/1080P.mp4"
DNN = "TF"
conf_threshold = 0.5


if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)

camera = cv2.VideoCapture(videoFile)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

grabbed = True
i = 0
while(camera.isOpened()):
    (grabbed, img) = camera.read()

    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
 
    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * width)
            y1 = int(detections[0, 0, i, 4] * height)
            x2 = int(detections[0, 0, i, 5] * width)
            y2 = int(detections[0, 0, i, 6] * height)

            cv2.rectangle( img, (x1,y1),(x2,y2),(0,255,0),2)

    cv2.imshow("Frame", imutils.resize(img, width=800))
    cv2.waitKey(1)
