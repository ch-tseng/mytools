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
import time

classList = { 1:"close", 2:"first", 3:"stretch", 4:"two" }
model_path = "/media/sf_VMshare/hand_graph/v2/frozen_inference_graph.pb"
pbtxt_path = "/media/sf_VMshare/hand_graph/v2/dnn_graph_v2.pbtxt"
resizeImg = (300,300)   #width for model
#source = "0"  # "0","1".. --> webcam, or "/xx/xxxx.mp4"
source = "/media/sf_VMshare/hand1.MOV"

make_video = False
make_video_path = "/media/sf_VMshare/out_hand1_v2.avi"
rotate = 180
display_width = 600


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

num_frames = 0
# Start time
start = time.time()

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
                cv2.putText(orgImg,classList[int(class_id)] ,(int(box_x), int(box_y)),cv2.FONT_HERSHEY_SIMPLEX,(.001*image_width),(0, 0, 255), 3)

        cv2.imshow("TEST", imutils.resize(orgImg, width=display_width))
        if(make_video is True):
            videoout.write(orgImg)

        cv2.waitKey(1)
        num_frames += 1

    else:
        if make_video is True:
            videoout.release()

        # End time
        end = time.time()
        # Time elapsed

        seconds = end - start
        print ("Time taken : {0} seconds".format(seconds))
 
        # Calculate frames per second
        fps  = num_frames / seconds;
        print ("Estimated frames per second : {0}".format(fps))
