# -*- coding: utf-8 -*-
#pip3 install mtcnn
#https://github.com/ipazc/mtcnn

import os, glob
import cv2
import numpy as np
import imutils
from mtcnn.mtcnn import MTCNN

img_path = "K:/Images/history"
#img_path = "Q:/慶生會"
save_face_path = "K:/Images/mosaic_face2"
face_side_threshold = 300
resize_img = 300
pix2pix_img = True

# ex: /data/spitdoor1/static/202001/2020-01-01/07/  --> images located in the third layer
layer_count = 1
#1: K:\Images\history\{2020020201}\images.jpg
#2: K:\Images\history\{2020}\{02}\images.jpg
#3: K:\Images\history\{2020}\{02}\{10}\images.jpg

def do_mosaic(frame, x, y, w, h, neighbor=5):
    """
    马赛克的实现原理是把图像上某个像素点一定范围邻域内的所有点用邻域内左上像素点的颜色代替，这样可以模糊细节，但是可以保留大体的轮廓。
    :param frame: opencv frame
    :param int x :  马赛克左顶点
    :param int y:  马赛克右顶点
    :param int w:  马赛克宽
    :param int h:  马赛克高
    :param int neighbor:  马赛克每一块的宽
    """
    fh, fw = frame.shape[0], frame.shape[1]
    if (y + h > fh) or (x + w > fw):
        print("cannot mosaic.")
        return frame

    for i in range(0, h - neighbor, neighbor):  # 关键点0 减去neightbour 防止溢出
        for j in range(0, w - neighbor, neighbor):
            rect = [j + x, i + y, neighbor, neighbor]
            color = frame[i + y][j + x].tolist()  # 关键点1 tolist
            left_up = (rect[0], rect[1])
            right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素
            cv2.rectangle(frame, left_up, right_down, color, -1)

    print("frame:", frame.shape)
    return frame

def Pix2Pix_img(img1, img2):
    x1, y1 = img1.shape[1], img1.shape[0]
    x2, y2 = img2.shape[1], img2.shape[0]
    
    bg1 = np.zeros((resize_img, resize_img, 3), dtype = 'uint8')
    x1_start = int((resize_img-x1)/2)
    x1_end = x1_start + x1
    y1_start = int((resize_img-y1)/2)
    y1_end = y1_start + y1
    bg1[y1_start:y1_end, x1_start:x1_end] = img1

    bg2 = np.zeros((resize_img, resize_img, 3), dtype = 'uint8')
    x2_start = int((resize_img-x2)/2)
    x2_end = x2_start + x2
    y2_start = int((resize_img-y2)/2)
    y2_end = y2_start + y2
    bg2[y2_start:y2_end, x2_start:x2_end] = img2

    vis = np.concatenate((bg1, bg2), axis=1)

    return vis

def get_dirs(folder_path, filter_txt=""):
    dirs = []
    for dir in os.listdir(folder_path):
        if(len(filter_txt)>0):
            if(dir[:len(filter_txt)] == filter_txt):
                dir_path = os.path.join(folder_path, dir)
                if(os.path.isdir(dir_path)):
                    dirs.append(dir_path)
        else:
            dir_path = os.path.join(folder_path, dir)
            if(os.path.isdir(dir_path)):
                dirs.append(dir_path)

    return dirs

def get_sorted_files(file_path):
    files = list(filter(os.path.isfile, glob.glob(os.path.join(file_path, "*.png"))))
    files.sort(key=lambda x: os.path.getmtime(x))
    return files

def getFaces(img, thresh=90):
    faces = detector.detect_faces(img)
    bboxes = []
    for face in faces:
        x = face["box"][0]
        y = face["box"][1]
        w = face["box"][2]
        h = face["box"][3]

        if(w>thresh and h>thresh):
            bboxes.append((x,y,w,h))

    return bboxes

def save_face(image, target_path, th_length):
    faces = getFaces(image, th_length)

    for (x,y,w,h) in faces:
        face_img = image[y:y+h, x:x+w]
        if(face_img.shape[1]>face_img.shape[0]):
            face_img = imutils.resize(face_img, width=resize_img)
        else:
            face_img = imutils.resize(face_img, height=resize_img)

        if(pix2pix_img is True):
            face_mosaic = do_mosaic(face_img.copy(), 35, 50, face_img.shape[1]-35, face_img.shape[0]-50, neighbor=11)
            img_pix2pix = Pix2Pix_img(face_mosaic, face_img)

        print("write to ", target_path)
        try:
            if(pix2pix_img is True):
                cv2.imwrite(target_path, img_pix2pix)
            else:
                cv2.imwrite(target_path, face_img)
        except:
            continue

def save_folder_files(dir_folder, th_length):
    img_files = get_sorted_files(dir_folder)
    print("images:", img_files)
    for img_file in img_files:
        img_filename = os.path.basename(img_file)
        print(img_file)
        save_face(cv2.imread(img_file), os.path.join(save_face_path,img_filename ), th_length)

if __name__ == "__main__":
    #layer 1
    detector = MTCNN()
    #dirs = get_dirs(img_path, filter_txt="2020")
    dirs = get_dirs(img_path)
    for dir_2 in dirs:
        #layer 2
        if(layer_count==1):
            try:
                save_folder_files(dir_2, face_side_threshold)
            except:
                continue
        else:    
            dirs_3 = get_dirs(dir_2)
            for dir_4 in dirs_3:
                if(layer_count==2):
                    save_folder_files(dir_4, face_side_threshold)
                else:
                    dirs_5 = get_dirs(dir_4)
                    for dir_6 in dirs_5:
                        if(layer_count==3):
                            save_folder_files(dir_6, face_side_threshold)
                        else:
                            dirs_7=get_dirs(dir_6)
                            for list_folder in dirs_7:
                                #print(dir)
                                save_folder_files(list_folder, face_side_threshold)


'''
detector = MTCNN()
pic = cv2.imread("peoples2.jpg")
faces = getFaces(pic)

for (x,y,w,h) in faces:
    cv2.rectangle( pic,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("TEST", pic)
cv2.waitKey(0)

'''
