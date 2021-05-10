from libDNNYolo import opencvYOLO
import cv2
import imutils
import time
import random


yolomodel = opencvYOLO( \
    mtype='yolov5', imgsize=(640,640), \
    objnames='models/yolov5/obj.names', \
    weights='models/yolov5/yolov5s_bodyparts.pt', \
    darknetcfg='', score=0.25, nms=0.55, gpu=True)
'''
yolomodel = opencvYOLO( \
    mtype='darknet', imgsize=(608,608), \
    objnames='models/yolov3/obj.names', \
    weights='models/yolov3/yolov3_last.weights', \
    darknetcfg='models/yolov3/yolov3.cfg', \
    score=0.25, nms=0.55, gpu=True)
'''
media = "../../modelSale/Himant/demo.mp4"
write_video = False
video_out = "demo.avi"

start = time.time()
last_time = time.time()
last_frames = 0


last_time, last_frames, fps = time.time(), 0, 0

def fps_count(total_frames):
    global last_time, last_frames, fps

    timenow = time.time()

    if(timenow - last_time)>6:
        fps  = (total_frames - last_frames) / (timenow - last_time)
        last_time  = timenow
        last_frames = total_frames

    return fps


def drawbox(img, box, txt, color):
    cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), color, 2)
    #cv2.putText(img, txt, (box[0], box[1]), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)

    return img

if __name__ == "__main__":

    print('Push Q to quit the program.')

    bcolors = [(255,255,0), (0,255,255), (255,0,0), (255,0,255) ]  #body, head

    INPUT = cv2.VideoCapture(media)

    frameID = 0
    if(video_out!=""):
        width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
        height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float

        if(write_video is True):
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            out = cv2.VideoWriter(video_out,fourcc, 24.0, (int(width),int(height)))

    hasFrame, frame = INPUT.read()
    while hasFrame:
        img = frame.copy()
        frameID += 1
        img = yolomodel.getObject(img, 0.25, 0.55, drawBox=True, bold=1, \
                    textsize=0.6, bcolor=(0,0,255), tcolor=bcolors, char_type='Chinese')

        cv2.imshow('test', imutils.resize(img, height=800))
        k = cv2.waitKey(1)
        if(k==113):
            break

        if write_video is True:
            out.write(img)

        hasFrame, frame = INPUT.read()

        fps = fps_count(frameID)
        print('FPS', fps)

    if write_video is True: out.release()
