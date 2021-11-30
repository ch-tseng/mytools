from libDNNYolo import opencvYOLO
import cv2
import imutils
import time, os, glob
import random
import numpy as np
from configparser import ConfigParser
import ast
from PIL import Image

cfg = ConfigParser()
cfg.read("demo.ini",encoding="utf-8")

#medias path for detection
media = cfg.get("Media", "media")
folder_layers = cfg.getint("Media", "folder_layers")
display_width = cfg.getint("Media", "display_width")
classes = ast.literal_eval(cfg.get("Model", "classes"))
min_face_size = ast.literal_eval(cfg.get("Media", "min_face_size"))
output_folder = cfg.get("Media", "output_folder")

#model configuration
model_size = ast.literal_eval(cfg.get("Model", "model_size"))
path_objname = cfg.get("Model", "path_objname")
path_weights = cfg.get("Model", "path_weights")
path_darknetcfg = cfg.get("Model", "path_darknetcfg")
score = float(cfg.get("Model", "confidence"))
nms = float(cfg.get("Model", "nms"))
gpu = cfg.getboolean("Model", "gpu")
tcolors = ast.literal_eval(cfg.get("Model", "tcolors"))

#----------------------------------------------------------------------
resize_video = None

media = media.replace('\\', '/')
path_objname = path_objname.replace('\\', '/')
path_weights = path_weights.replace('\\', '/')
path_darknetcfg = path_darknetcfg.replace('\\', '/')

if path_weights[-2:] == 'pt':
    face_model = opencvYOLO( \
        mtype='yolov5', imgsize=model_size, \
        objnames=path_objname, \
        weights=path_weights, \
        darknetcfg='', score=score, nms=nms, tcolors=tcolors, gpu=gpu)
else:
    face_model = opencvYOLO( \
        mtype='darknet', imgsize=model_size, \
        objnames=path_objname, \
        weights=path_weights, \
        darknetcfg=path_darknetcfg, \
        score=score, nms=nms, tcolors=tcolors, gpu=gpu)


if os.path.isdir(media):
    img_list = []

    if folder_layers == 2:
        for folder in os.listdir(media):

            folder_path = os.path.join(media, folder)

            if os.path.isdir(folder_path):

                for imgext in ['jpg','jpeg','JPG','png','PNG']:
                    img_path_name = os.path.join(folder_path,  '*.'+imgext)

                    img_list += glob.glob( img_path_name)

    else:
        for imgext in ['jpg','jpeg','JPG','png','PNG']:
            img_list += glob.glob(os.path.join(media, '*.'+imgext))

    media_type = 'imgs'
    INPUT = img_list
else:
    media_type = 'video'
    INPUT = cv2.VideoCapture(media)

start = time.time()
last_time = time.time()
last_frames = 0

last_time, last_frames, fps = time.time(), 0, 0

def get_img():
    global frameID

    frame = None
    if media_type == 'video':
        hasFrame, frame = INPUT.read()
        frame_name = None
        if resize_video is not None:
            frame = cv2.resize(frame, resize_video)

    else:
        if frameID>=len(INPUT):
            hasFrame = False
            frame_name = None

        else:

            hasFrame = True
            img = Image.open(INPUT[frameID])
            img=np.array(img)  
            frame=cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img_path = INPUT[frameID].replace('\\', '/')    
            print(img_path)
            frame_name = img_path.split('/')[-2]
            
            #frameID += 1

    return hasFrame, frame, frame_name

def fps_count(total_frames):
    global last_time, last_frames, fps

    timenow = time.time()

    if(timenow - last_time)>6:
        fps  = (total_frames - last_frames) / (timenow - last_time)
        last_time  = timenow
        last_frames = total_frames

    return fps

def get_face(img):
     
    srcimg = img.copy()

    img = face_model.getObject(img, score, nms, drawBox=True, char_type='Chinese')
    #cv2.imshow('test', imutils.resize(img, width=display_width))
    
    maxID, maxArea = None, 0
    if len(face_model.bbox)>0:        

        for id, (x,y,w,h) in enumerate(face_model.bbox):
            if w*h > maxArea:
                if w>min_face_size[0] and h>min_face_size[1]:
                    maxID = id

    face_area = None
    if maxID is not None:
        (xx,yy,ww,hh) = face_model.bbox[maxID]
        if xx<0: xx=0
        if yy<0: yy=0

        face_area = srcimg[yy:yy+hh, xx:xx+ww]
        #cv2.imshow('face', face_area)

    #cv2.waitKey(1)

    return face_area


if __name__ == "__main__":

    print('Push Q to quit the program.')
    if not os.path.exists('output'):
        os.makedirs('output')

    frameID = 0
    hasFrame, frame, frame_name = get_img()

    while hasFrame:

        face_area = get_face(frame)
        if face_area is None:
            continue

        if folder_layers == 2:
            path_folder = os.path.join(output_folder, frame_name)            
            if not os.path.exists(path_folder):
                os.makedirs(path_folder)

            img_path_write = os.path.join(path_folder, str(time.time())+'.jpg')
            img_path_write = img_path_write.replace('\\', '/')
            print('output', img_path_write)

        else:
            img_path_write = os.path.join(output_folder, str(time.time())+'.jpg')

        #cv2.imwrite( img_path_write, face_area)
        img_pil = Image.fromarray(cv2.cvtColor(face_area,cv2.COLOR_BGR2RGB))
        img_pil.save(img_path_write)

        '''
        if media_type == 'video':
            k = cv2.waitKey(1)
        else:
            k = cv2.waitKey(1)

        if(k==113):
            break
        '''
        hasFrame, frame, frame_name = get_img()
        frameID += 1

        fps = fps_count(frameID)
        print('FPS', fps)

