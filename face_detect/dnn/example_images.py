import cv2
import imutils
import time, glob, os

model_img_size = 360
img_folder = "D:/works/peoples_male_female/male/*"
DNN = "CAFFE"
conf_threshold = 0.5


if DNN == "CAFFE":
    modelFile = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
    configFile = "deploy.prototxt"
    net = cv2.dnn.readNetFromCaffe(configFile, modelFile)
else:
    modelFile = "opencv_face_detector_uint8.pb"
    configFile = "opencv_face_detector.pbtxt"
    net = cv2.dnn.readNetFromTensorflow(modelFile, configFile)



i = 0
for file in glob.glob(img_folder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension in ['.jpg', '.png', '.jpeg', '.bmp']):
        img = cv2.imread(file)
        try:
            width, height = img.shape[1], img.shape[0]

        except:
            print('file error:', file)
            continue

        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), [104, 117, 123], False, False)
        net.setInput(blob)
        detections = net.forward()
        faces, boxes = [], []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)

            if(x2-x1>model_img_size or y2-y1>model_img_size):
                faces.append(img[y1:y2, x1:x2])
                boxes.append([x1,x2,y1,y2])

            cv2.rectangle( img, (x1,y1),(x2,y2),(0,255,0),2)

        cv2.imshow("Frame", imutils.resize(img, width=800))
        cv2.waitKey(0)
