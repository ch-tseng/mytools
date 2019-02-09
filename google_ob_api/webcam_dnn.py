#cd ~/works/opencv/samples/dnn
#python tf_text_graph_ssd.py \ 
#    --input /media/sf_VMshare/Mobilenet_paint_on_air/graph/frozen_inference_graph.pb \
#    --output frozen_paint_on_air.pbtxt  \
#    --config /media/sf_VMshare/Mobilenet_paint_on_air/graph/pipeline.config

#python tf_text_graph_faster_rcnn.py \
#    --input /media/sf_VMshare/palm_num/faster_rcnn/graph/frozen_inference_graph.pb \
#    --config /media/sf_VMshare/palm_num/faster_rcnn/training/pipeline.config \
#    --output /media/sf_VMshare/palm_num/faster_rcnn/dnn/graph.pbtxt

import cv2
import imutils

model_path = "/media/sf_mobileNet/paint_on_air/frozen_inference_graph_456826.pb"
pbtxt_path = "/media/sf_mobileNet/paint_on_air/dnn_paint_on_air.pbtxt"
resizeImg = (300,300)   #width for model
#source = "0"  # "0","1".. --> webcam, or "/xx/xxxx.mp4"
source = "/media/sf_mobileNet/paint_on_air/IMG_0549.MOV"

make_video = True
make_video_path = "/media/sf_mobileNet/paint_on_air/dnn_paint_on_air.avi"
rotate = 90
display_width = 450


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
        if(rotate != 0):
            frame = imutils.rotate_bound(frame, rotate)

        orgImg = frame.copy()

        model.setInput(cv2.dnn.blobFromImage(frame, size=resizeImg, swapRB=True))
        output = model.forward()
        output[0,0,:,:].shape is (100, 7)

        for detection in output[0, 0, :, :]:
            confidence = detection[2]
            if confidence > .3:
                class_id = detection[1]
                #print(str(str(class_id) + " " + str(detection[2])))
                print(detection)

                image_height, image_width, _ = frame.shape
                box_x=detection[3] * image_width
                box_y=detection[4] * image_height
                box_width=detection[5] * image_width
                box_height=detection[6] * image_height

                cv2.rectangle(orgImg, (int(box_x), int(box_y)), (int(box_width), int(box_height)), (0, 255, 0), thickness=3)
                cv2.putText(orgImg,str(int(class_id)) ,(int(box_x), int(box_y)),cv2.FONT_HERSHEY_SIMPLEX,(.002*image_width),(0, 0, 255), 3)

        cv2.imshow("TEST", imutils.resize(orgImg, width=display_width))
        videoout.write(orgImg)
        cv2.waitKey(1)

    else:
        if make_video is True:
            videoout.release()
