#http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

import glob, os
import os.path
import time
from shutil import copyfile
import cv2
#from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

#--------------------------------------------------------------------
annoFilePath = "/media/sf_VMshare/2yolo/list_bbox_celeba.txt"
imgFolder = "/media/sf_VMshare/2yolo/images/"
saveYoloPath = "/media/sf_VMshare/2yolo/yolo/"
classID = ("face", 0)

#---------------------------------------------------------------------

if not os.path.exists(saveYoloPath):
    os.makedirs(saveYoloPath)

def transferYolo( img_name, bbox):
    img_filename, _ = img_name.split('.')
    full_file = os.path.join(imgFolder, img_name)

    x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]

    if(os.path.exists(full_file)):
        img = cv2.imread(full_file)
        imgShape = img.shape
        print (full_file + '-->', img.shape)
        img_h = imgShape[0]
        img_w = imgShape[1]

        yoloFilename = os.path.join(saveYoloPath, img_filename + ".txt")
        print("     ............writeing to {}".format(yoloFilename))

        with open(yoloFilename, 'a') as the_file:
            xx = (x + (w/2)) * 1.0 / img_w
            yy = (y + (h/2)) * 1.0 / img_h
            ww = (w * 1.0) / img_w
            hh = (h * 1.0) / img_h

            the_file.write(img_filename+'.txt' + ' ' + str(xx) + ' ' + str(yy) + ' ' + str(ww) + ' ' + str(hh) + '\n')


        the_file.close()

#---------------------------------------------------------------
fileCount = 0
anno_file = open(annoFilePath, 'r')

annotations = []
for line in anno_file:
    if("image_id" not in line):
        anno = []
        for x in line.split(' '):
            data = x.strip()
            if(len(data)>0):
                #print(data)
                anno.append(data)

        if(len(anno)==5):
            print(anno[0])
            transferYolo( anno[0], [int(anno[1]), int(anno[2]), int(anno[3]), int(anno[4]) ] )
            annotations.append(anno)

print(annotations)
