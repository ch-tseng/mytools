from libDNNYolo import opencvYOLO
import cv2
import imutils
import time
import random

media = r"D:\OneDrive\Model_Sale\crowd_human\crowd_humang\videos\demo.mp4"
write_video = False
video_out = "D:\OneDrive\Model_Sale\crowd_human\crowd_humang\videos\demo.avi"

yolo_type = 'yolov5'  #yolov5 or darknet
model_size = (640,640)
path_objname = r"D:\OneDrive\Model_Sale\crowd_human\crowd_humang\YOLO5_v2\yolo5s_640\obj.names"
path_weights = r"D:\OneDrive\Model_Sale\crowd_human\crowd_humang\YOLO5_v2\yolo5s_640\crowd_human.pt"
path_darknetcfg = r""
score = 0.25
nms = 0.55
gpu = False

#----------------------------------------------------------------------

media = media.replace('\\', '/')
video_out = video_out.replace('\\', '/')
path_objname = path_objname.replace('\\', '/')
path_weights = path_weights.replace('\\', '/')
path_darknetcfg = path_darknetcfg.replace('\\', '/')

if path_weights[-2:] == 'pt':
    yolomodel = opencvYOLO( \
        mtype='yolov5', imgsize=model_size, \
        objnames=path_objname, \
        weights=path_weights, \
        darknetcfg='', score=score, nms=nms, gpu=gpu)
else:
    yolomodel = opencvYOLO( \
        mtype='darknet', imgsize=model_size, \
        objnames=path_objname, \
        weights=path_weights, \
        darknetcfg=path_darknetcfg, \
        score=score, nms=nms, gpu=gpu)

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


if __name__ == "__main__":

    print('Push Q to quit the program.')

    #bcolors = [(255,255,0), (0,255,255), (255,0,0), (255,0,255) ]  #body, head

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
        img = yolomodel.getObject(img, 0.25, 0.55, drawBox=True, char_type='Chinese')

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
