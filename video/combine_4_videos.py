import cv2
import numpy as np
import imutils

#----------------------------------------------------------------
title1 = "EfficientDet D0"
title2 = "EfficientDet D1"
title3 = "YOLOV3"
title4 = "YOLOV4"
title_loc = (30, 50)
title_size = 1.2

video1 = "/DATA1/trained_models/EfficientDet/Vehicles/output/IMG_3512_D0.avi"
video2 = "/DATA1/trained_models/EfficientDet/Vehicles/output/IMG_3512_D2.avi"
video3 = "/DATA1/trained_models/EfficientDet/Vehicles/output/IMG_3512_yolov3.avi"
video4 = "/DATA1/trained_models/EfficientDet/Vehicles/output/IMG_3512_yolov4.avi"

resize_width, resize_height = 960, 540
video_rate = 15
#combine = "H"  #V: vertical , H: Horizontal
output = "/DATA1/trained_models/EfficientDet/Vehicles/output/combine_IMG_3512.avi"

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

camera4 = cv2.VideoCapture(video4)
width4 = int(camera4.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height4 = int(camera4.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("video 4 resolution is: %d x %d" % (width4, height4))


#if(combine=="H"):
w = resize_width * 2
h = resize_height * 2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output, fourcc, video_rate, (w, h))

grabbed1 = True

i = 0
while grabbed1:
    i += 1
    print("frame #", i)
    (grabbed1, frame1) = camera1.read()
    if(grabbed1 is not True):
        print("Video1 is finished.")
        break
    frame1 = cv2.resize(frame1, (resize_width, resize_height))
    cv2.putText(frame1, title1, title_loc, cv2.FONT_HERSHEY_SIMPLEX, title_size, (0,255,255), 3)

    (grabbed2, frame2) = camera2.read()
    if(grabbed2 is not True):
        print("Video2 is finished.")
        break
    frame2 = cv2.resize(frame2, (resize_width, resize_height))
    cv2.putText(frame2, title2, title_loc, cv2.FONT_HERSHEY_SIMPLEX, title_size, (0,255,255,), 3)

    (grabbed3, frame3) = camera3.read()
    if(grabbed3 is not True):
        print("Video3 is finished.")
        break
    frame3 = cv2.resize(frame3, (resize_width, resize_height))
    cv2.putText(frame3, title3, title_loc, cv2.FONT_HERSHEY_SIMPLEX, title_size, (255,255,0), 3)

    (grabbed4, frame4) = camera4.read()
    if(grabbed4 is not True):
        print("Video4 is finished.")
        break
    frame4 = cv2.resize(frame4, (resize_width, resize_height))
    cv2.putText(frame4, title4, title_loc, cv2.FONT_HERSHEY_SIMPLEX, title_size, (255,255,0), 3)

    video_combine1 = np.concatenate((frame1, frame2), axis=1)
    video_combine2 = np.concatenate((frame3, frame4), axis=1)
    video_combine_together = np.concatenate((video_combine1, video_combine2), axis=0)

    #cv2.imshow("Combined", imutils.resize(video_combine, height=350))
    #cv2.waitKey(1)
    out.write(video_combine_together)


print("Video combined")
