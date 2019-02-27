import cv2
import imutils

#----------------------------------------------------

cam_id = 0
write_output = True
output_video_path = "/media/sf_VMshare/output.avi"
video_size = (640, 480)  #x,y
video_rate = 24.0

#----------------------------------------------------

camera = cv2.VideoCapture(cam_id)
width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
print("This webcam's resolution is: %d x %d" % (width, height))

if(write_output is True):
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, video_size[0])
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, video_size[1])
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, video_rate, (int(width),int(height)))

    grabbed = True

    i = 0
    while grabbed:
        i += 1
        (grabbed, frame) = camera.read()

        if(write_output is True):
            print("write frame id #{} ({}x{}) to file".format(i, frame.shape[1], frame.shape[0]))
            out.write(frame)

        cv2.imshow("webcam id:"+str(cam_id), imutils.resize(frame, height=300))
        cv2.waitKey(1)

    out.release()
    camera.release()
    
