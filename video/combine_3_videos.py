import cv2
import numpy as np
import imutils

#----------------------------------------------------------------
title1 = "Old YOLOV5S"
title2 = "New YOLOV5S"
title3 = "New YOLOV5M"

video1 = "old.avi"
video2 = "yolov5s.avi"
video3 = "yolov5m.avi"

video_rate = 8
combine = "H"  #V: vertical , H: Horizontal
output = "combine.avi"

#-----------------------------------------------------------------
camera1 = cv2.VideoCapture(video1)
width1 = int(camera1.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height1 = int(camera1.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("video 1 resolution is: %d x %d" % (width1, height1))

camera2 = cv2.VideoCapture(video2)
width2 = int(camera2.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height2 = int(camera2.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("video 2 resolution is: %d x %d" % (width2, height2))

camera3 = cv2.VideoCapture(video3)
width3 = int(camera3.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height3 = int(camera3.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("video 3 resolution is: %d x %d" % (width3, height3))


if(combine=="H"):
    w = width1 + width2 + width3
    h = height1

else:
    w = width1
    h = height1 + height2 + height3


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output, fourcc, video_rate, (w, h))

grabbed1 = True

i = 0
while grabbed1:
    i += 1
    (grabbed1, frame1) = camera1.read()
    (grabbed2, frame2) = camera2.read()
    (grabbed3, frame3) = camera3.read()

    if(combine=="H"):
        video_combine = np.concatenate((frame1, frame2, frame3), axis=1)

    else:
        video_combine = np.concatenate((frame1, frame2, frame3), axis=0)

    cv2.putText(video_combine, title1, (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,0), 2)
    cv2.putText(video_combine, title2, (500, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,0), 2)
    cv2.putText(video_combine, title3, (980, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255,255,0), 2)
    cv2.imshow("Combined", imutils.resize(video_combine, height=640))
    cv2.waitKey(1)
    out.write(video_combine)


print("Video combined")
