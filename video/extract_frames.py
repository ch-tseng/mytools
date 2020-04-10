import cv2
import imutils
import os

#----------------------------------------------------

videos = [ 
    #       "/media/sf_VMshare/money/IMG_2977.MOV",
    #       "/media/sf_VMshare/money/IMG_2981.MOV",
    #        "/media/sf_VMshare/money/IMG_2978.MOV",
    #        "/media/sf_VMshare/money/IMG_2980.MOV",
    #        "/media/sf_VMshare/money/IMG_2982.MOV"
    #        "/media/sf_VMshare/money/IMG_2984.MOV",
    #        "/media/sf_VMshare/money/IMG_2986.MOV"
    #        "/media/sf_VMshare/money/IMG_2985.MOV",
    #        "/media/sf_VMshare/money/IMG_2987.MOV"
    #        "/media/sf_VMshare/money/IMG_2992.MOV"
            "J:/iKlips Camera/CameraCapture_20200409174211_2.MOV"

         ]
output_frame_path = "K:/Datatsets/waiting/CameraCapture_20200409174211_2"
resize_output = False
output_resize_ratio = 0.75
rotate_img = False
rotate = 0
interval_frames = 30
prefix_char = "a_"
#----------------------------------------------------

if not os.path.exists(output_frame_path):
    os.makedirs(output_frame_path)

i = 0
I = 0
for video_path in videos:
    camera = cv2.VideoCapture(video_path)
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
    print("This video's resolution is: %d x %d" % (width, height))

    grabbed = True

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
                cv2.imwrite(output_frame_path + "/" + prefix_char + str(I).rjust(6,"0") + ".jpg" , frame)

    camera.release()
