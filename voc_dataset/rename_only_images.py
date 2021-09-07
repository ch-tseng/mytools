# -*- coding: utf-8 -*-
import cv2
import os
import imutils
from tqdm import tqdm

dataset_images = r"G:\For_Eden_KS_road\20210719"
extract_to = r"P:\to_ritchie\20210719"
name_format = 'B00000000.jpg'

img_resize = True
resize_width = 1920

dataset_images = dataset_images.replace('\\', '/')
extract_to = extract_to.replace('\\', '/')

def chkEnv():
    if not os.path.exists(extract_to):
        os.makedirs(extract_to)
        print("no {} folder, created.".format(extract_to))

chkEnv()

nformat = name_format.split('.')
pad_count = len(nformat[0])-1
ext_name = nformat[1]

id = 0
errid = 0

for file in tqdm(os.listdir(dataset_images)):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        file_path = os.path.join(dataset_images, file)
        img = cv2.imread(file_path)
        #print(file)
        try:
            test = img.shape
        except:
            errid += 1
            print('#{} - ERROR: {} is error, cannot read.'.format(errid, file_path))

            continue

        id += 1
        new_name = name_format[:1] + str(id).zfill(pad_count) + '.' + ext_name
        newpath = os.path.join(extract_to, new_name)

        if img_resize is True:
            h ,w = img.shape[0], img.shape[1]
            if h>w:
                img = imutils.resize(img, height = resize_width)
            else:
                img = imutils.resize(img, width = resize_width)

        cv2.imwrite(newpath, img)