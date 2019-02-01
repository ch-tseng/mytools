#cd ~/works/opencv/samples/dnn
#python tf_text_graph_ssd.py --input /media/sf_VMshare/Mobilenet_paint_on_air/graph/frozen_inference_graph.pb --output frozen_paint_on_air.pbtxt --config /media/sf_VMshare/Mobilenet_paint_on_air/graph/pipeline.config
#python tf_text_graph_faster_rcnn.py --input /media/sf_VMshare/palm_num/faster_rcnn/graph/frozen_inference_graph.pb --config /media/sf_VMshare/palm_num/faster_rcnn/training/pipeline.config --output /media/sf_VMshare/palm_num/faster_rcnn/dnn/graph.pbtxt

import cv2
import imutils

model_path = "/media/sf_VMshare/palm_num/faster_rcnn/graph/frozen_inference_graph.pb"
pbtxt_path = "/media/sf_VMshare/palm_num/faster_rcnn/dnn/graph.pbtxt"
resizeImg = 300   #width
source = "0"  # "0","1".. --> webcam, or "/xx/xxxx.mp4"

make_video = True
make_video_path = "/media/sf_VMshare/palm_num_faster_rcnn.avi"


#----------------------------------------------------------------------------
model = cv2.dnn.readNetFromTensorflow(model_path, pbtxt_path)

if(len(source)==1):
    camera = cv2.VideoCapture(int(source))
else:
    camera = cv2.VideoCapture(source)

width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

if make_video is True:
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    videoout = cv2.VideoWriter(make_video_path, fourcc, 30, (width, height))

grabbed = True

def id_class_name(class_id, classes):
    for key,value in classes.items():
        if class_id == key:
            return value

while grabbed:

    (grabbed, frame) = camera.read()
    if(grabbed is True):
        orgImg = frame.copy()
        frame = imutils.resize(frame, width=resizeImg)

        model.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True))
        output = model.forward()
        output[0,0,:,:].shape is (100, 7)

        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > .3:
                class_id = detection[1]
                print(str(str(class_id) + " " + str(detection[2])))

                image_height, image_width, _ = frame.shape
                box_x=detection[3] * width
                box_y=detection[4] * height
                box_width=detection[5] * width
                box_height=detection[6] * height

                cv2.rectangle(orgImg, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (23, 230, 210), thickness=1)
                cv2.putText(orgImg,str(class_id) ,(int(box_x), int(box_y+.05*image_height)),cv2.FONT_HERSHEY_SIMPLEX,(.005*image_width),(0, 0, 255))

        cv2.imshow("TEST", orgImg)
        videoout.write(orgImg)
        cv2.waitKey(1)

    else:
        videoout.release()

