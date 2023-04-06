import cv2
import imutils
from datetime import datetime, timedelta
#----------------------------------------------------

media = r"e:\temp\IMG_2852.MOV"
output_video_path = r"e:\temp\output.avi"
datetime_start = "2023/04/06 14:06:05"
write_output = True

#----------------------------------------------------

media = media.replace('\\', '/')
output_video_path = output_video_path.replace('\\', '/')

camera = cv2.VideoCapture(media)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
fps = camera.get(cv2.CAP_PROP_FPS)
timestamps = [camera.get(cv2.CAP_PROP_POS_MSEC)]
calc_timestamps = [0.0]
    
print("This webcam's resolution is: %d x %d (FPS:%d)" % (width, height, fps))

datetime_object = datetime.strptime(datetime_start, '%Y/%m/%d %H:%M:%S')

if(write_output is True):
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (int(width),int(height)))

    grabbed = True

    i = 0
    time_diff = 0
    (grabbed, frame) = camera.read()
    while grabbed:
        i += 1
        timestamps.append(camera.get(cv2.CAP_PROP_POS_MSEC))
        time_diff += calc_timestamps[-1] + 1000/fps
        time_now = datetime_object + timedelta(seconds=( int(time_diff/1000)))
        cv2.putText(frame, "{}".format(time_now), (frame.shape[1]-850,frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0,255,0),2)
        print(time_now)

        if(write_output is True):
            #print("write frame id #{} ({}x{}) to file".format(i, frame.shape[1], frame.shape[0]))
            out.write(frame)

        cv2.imshow("Frame", imutils.resize(frame, height=300))
        cv2.waitKey(1)
        (grabbed, frame) = camera.read()
        
    out.release()
    camera.release()
    
