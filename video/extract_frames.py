import cv2
import imutils
import os

#----------------------------------------------------

video_path = "/media/sf_VMshare/money/IMG_2977.MOV"
output_frame_path = "/media/sf_VMshare/money/images/1"
resize_output = False
output_resize_ratio = 0.75
rotate_img = False
rotate = 90
interval_frames = 15

#----------------------------------------------------

if not os.path.exists(output_frame_path):
    os.makedirs(output_frame_path)


camera = cv2.VideoCapture(video_path)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("This video's resolution is: %d x %d" % (width, height))

grabbed = True

i = 0
I = 0
while grabbed:
    i += 1
    (grabbed, frame) = camera.read()

    if(grabbed is True): 

        if(resize_output is True):
            frame = imutils.resize(frame, width=int(width*output_resize_ratio), height=int(height**output_resize_ratio))

        if(rotate_img is True):
            frame = imutils.rotate_bound(frame, rotate)

        cv2.imshow("Frame", imutils.resize(frame, height=300))
        cv2.waitKey(1)

        if(i % interval_frames == 0):
            I += 1
            cv2.imwrite(output_frame_path + "/" + str(I).rjust(6,"0") + ".jpg" , frame)

    else:
        camera.release()
