from yoloPydarknet import pydarknetYOLO
import cv2
import imutils
import time

yolo = pydarknetYOLO(obdata="../darknet/cfg/coco.data", weights="yolov3.weights", 
    cfg="../darknet/cfg/yolov3.cfg")
video_out = "yolo_output.avi"

start_time = time.time()

if __name__ == "__main__":

    VIDEO_IN = cv2.VideoCapture(0)
        if(video_out!=""):
            width = int(VIDEO_IN.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            height = int(VIDEO_IN.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(video_out,fourcc, 30.0, (int(width),int(height)))

    frameID = 0
    while True:
        hasFrame, frame = VIDEO_IN.read()
        # Stop the program if reached end of video
        if not hasFrame:
            print("Done processing !!!")
            print("--- %s seconds ---" % (time.time() - start_time))
            break

        yolo.getObject(frame, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
        print ("Object counts:", yolo.objCounts)
        cv2.imshow("Frame", imutils.resize(frame, width=850))
        if(video_out!=""):
            out.write(frame)

        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            out.release()
            break
