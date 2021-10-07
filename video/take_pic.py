import cv2
import imutils
import sys, os, time
from datetime import datetime

#----------------------------------------------------

record_type = 'video'  #image, video
cam_id = 0
output_path = "pic_takes/"
video_size = (1920, 1080)  #x,y
video_rate = 5.0

#----------------------------------------------------

def exit_app():
    global wout

    camera.release()
    if wout is not None: wout.release()
    sys.exit()

def fps_count(total_frames):
    global last_time, last_frames

    timenow = time.time()
    fps = fpsnow

    if(timenow - last_time)>5:
        fps  = (total_frames - last_frames) / (timenow - last_time)
        print("FPS: {0}".format(fps))

        last_time  = timenow
        last_frames = total_frames

    return fps

def display_img(img):
    img = cv2.resize(img, (1280, 720))
    cv2.putText(img, '{}x{} FPS:{}'.format(width, height, round(fpsnow, 1)), ( img.shape[1]-340, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  (0,255,0), 2, cv2.LINE_AA)
    cv2.putText(img, datenow, (img.shape[1]-320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,  (0,255,0), 1, cv2.LINE_AA)
    cv2.putText(img, 'images:{}'.format(i), (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  (0,255,0), 2, cv2.LINE_AA)
    if write_now is True:
        cv2.putText(img, 'recording...', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0,  (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("dst", img)


def new_video(id):
    filename = datetime.now().strftime('%Y%m%d%H%M%S') + '_{}.avi'.format(id)
    output_video_path = os.path.join(output_path, filename)

    out = cv2.VideoWriter(output_video_path, fourcc, video_rate, (int(width),int(height)))

    return out

#cv2.namedWindow("dst", cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty("dst",cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
(a,b,screenWidth,screenHeight) = cv2.getWindowImageRect('dst')
print("LCD's resolution is: %d x %d" % (screenWidth, screenHeight))

camera = cv2.VideoCapture(cam_id)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
camera.set(cv2.CAP_PROP_FPS, 30)

fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')

print("USB Camera's resolution is: %d x %d" % (width, height))

if not os.path.exists(output_path):
    os.makedirs(output_path)

write_now = False
i = 0
fid = 1
start = time.time()
timenow = time.time()
record_frames = 0
wout = None
vid = 0
last_time = 0
last_frames = 0
fpsnow = 0

(grabbed, frame) = camera.read()
if __name__ == '__main__':
    while grabbed:
        datenow = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        display_img(frame.copy())
        k = cv2.waitKey(1)

        if k==113:
            exit_app()

        elif k == 32:
            if write_now is False:
                write_now = True
            else:
                write_now = False

        if write_now is True:
            if record_type == 'image':
                datenow_string = datenow.replace(' ','_')
                datenow_string = datenow_string.replace(':','')
                img_path = os.path.join(output_path, '{}_{}.jpg'.format(i,datenow_string))
                i += 1
                cv2.imwrite(img_path, frame)

            else:
                if wout is None: wout = new_video(vid)
                
                if (record_frames > 1000):
                    if vid>0:                        
                        wout.release()

                    vid += 1
                    record_frames = 0
                    wout = new_video(vid)

                wout.write(frame)
                record_frames += 1


        (grabbed, frame) = camera.read()
        fpsnow = fps_count(fid)
        fid += 1

    out.release()
    if wout is not None: wout.release()

    camera.release()
    
