from yoloOpencv import opencvYOLO
import cv2
import imutils
import time

yolo = opencvYOLO(modeltype="yolov3", imgsize=(1024,1024), \
    objnames="crowd_humans/obj.names", \
    weights="crowd_humans/yolov3_500000.weights",\
    cfg="crowd_humans/yolov3.cfg")

inputType = "image"  # webcam, image, video
media = "../../detect_small_peoples/group_photo.jpg"
write_video = False
video_out = "/media/sf_VMshare/out_hand1_yolov3.avi"
output_rotate = True
rotate = 0

start_time = time.time()

def findObject(frame):
    yolo.getObject(frame, labelWant="", drawBox=False, bold=2, textsize=1.2, bcolor=(0,255,0), tcolor=(0,0,255))
    #print(yolo.classIds, yolo.bbox, yolo.scores)
    return yolo.classIds, yolo.bbox, yolo.scores

if __name__ == "__main__":

    if(inputType == "webcam"):
        INPUT = cv2.VideoCapture(0)
    elif(inputType == "image"):
        INPUT = cv2.imread(media)
        INPUT = imutils.resize(INPUT, width=1024)
    elif(inputType == "video"):
        INPUT = cv2.VideoCapture(media)

    total_counts = 0
    if(inputType == "image"):
        classes, bboxes, scores = findObject(INPUT)
        for id, (xf,yf,wf,hf) in enumerate(bboxes):
            if(classes[id] == 0):
                cv2.rectangle( INPUT,(xf,yf),(xf+wf,yf+hf),(0,255,0),2)
                total_counts += 1

        cv2.putText(INPUT, "People counting:{}".format(total_counts), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0),2)
        cv2.imwrite("output.jpg", INPUT)

        cv2.imshow("Frame", imutils.resize(INPUT, width=850))
        k = cv2.waitKey(0)
        if k == 0xFF & ord("q"):
            out.release()

    else:
        if(video_out!=""):
            width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

            if(write_video is True):
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(video_out,fourcc, 30.0, (int(width),int(height)))

        frameID = 0

        while True:
            hasFrame, frame = INPUT.read()
            # Stop the program if reached end of video
            if not hasFrame:
                print("Done processing !!!")
                # End time
                end = time.time()
                # Time elapsed
                seconds = end - start_time
                print ("Time taken : {0} seconds".format(seconds))
 
                # Calculate frames per second
                fps  = frameID / seconds;
                print ("Estimated frames per second : {0}".format(fps))
                break

            if(output_rotate is True):
                frame = imutils.rotate(frame, rotate)

            yolo.getObject(frame, labelWant="", drawBox=False, bold=5, textsize=1.2, bcolor=(0,255,0), tcolor=(0,0,255))
            print ("Object counts:", yolo.objCounts)
            #yolo.listLabels()
            print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
                format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
            cv2.imshow("Frame", imutils.resize(frame, width=850))
            frameID += 1

            if(write_video is True):
                out.write(frame)

            k = cv2.waitKey(1)
            if k == 0xFF & ord("q"):
                if(write_video is True):
                    out.release()

                break
