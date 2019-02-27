import cv2
import numpy as np
import imutils

#----------------------------------------------------------------
title1 = "SSD-MobileNet V2"
title2 = "YOLOV3-Tiny"

video1 = "/media/sf_VMshare/out_hand1_v2.avi"
video2 = "/media/sf_VMshare/out_hand1_yolov3_tiny.avi"

video_rate = 20
combine = "H"  #V: vertical , H: Horizontal
output = "/media/sf_VMshare/output_combine2.avi"

#-----------------------------------------------------------------
camera1 = cv2.VideoCapture(video1)
width1 = int(camera1.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height1 = int(camera1.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("video 1 resolution is: %d x %d" % (width1, height1))

camera2 = cv2.VideoCapture(video2)
width2 = int(camera2.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height2 = int(camera2.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("video 2 resolution is: %d x %d" % (width2, height2))

if(combine=="H"):
    w = width1 + width2
    h = height1

else:
    w = width1
    h = height1 + height2


fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output, fourcc, video_rate, (w, h))

grabbed1 = True

i = 0
while grabbed1:
    i += 1
    (grabbed1, frame1) = camera1.read()
    (grabbed2, frame2) = camera2.read()

    if(combine=="H"):
        video_combine = np.concatenate((frame1, frame2), axis=1)

    else:
        video_combine = np.concatenate((video1, video2), axis=0)

    cv2.putText(video_combine, title1, (250, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)
    cv2.putText(video_combine, title2, (1550, 90), cv2.FONT_HERSHEY_SIMPLEX, 3, (255,0,0), 3)
    cv2.imshow("Combined", imutils.resize(video_combine, height=350))
    cv2.waitKey(1)
    out.write(video_combine)


print("Video combined")
