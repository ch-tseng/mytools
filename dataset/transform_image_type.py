#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import shutil
from PIL import Image, ImageDraw, ImageFont #dynamic import

ds_folder = "/home/digits/datasets/Diabnext_277_classes_org2"
#transfer these image types
image_types = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
#to target image types
to_image_type = '.png'
IMAGE_SHAPE = (300,300)

def gif_trans(base_name, gif, loc_path):
    img = Image.open(gif).convert('RGB')
    #imgType = to_image_type[1:]
    print("transform {} to {}".format(gif, os.path.join(loc_path, 'gif-'+base_name+to_image_type)))
    img.save(os.path.join(loc_path, 'gif-'+base_name+to_image_type))

for classname in os.listdir(ds_folder):
    folder_1  = os.path.join(ds_folder, classname)

    if(os.path.isdir(folder_1)):
        i = 0
        for subfolder in os.listdir(folder_1):
            file  = os.path.join(folder_1, subfolder)

            if(os.path.isdir(file)):
                shutil.rmtree(file)
            else:
                filename, file_extension = os.path.splitext(file)

                if( file_extension.lower() not in image_types):
                    print("remove file:", file)
                    os.remove(file)
                else:
                    if(file_extension.lower() == '.gif'):
                        file_name_no_ext = os.path.basename(filename)
                        gif_trans(file_name_no_ext, file, folder_1)
                        os.remove(file)
                        print("GIF: {}) {} transformed.".format(i, os.path.join(folder_1, filename+to_image_type)))

                    elif(file_extension.lower() != to_image_type):
                        img = cv2.imread(file)
                        try:
                            img = cv2.resize(img,IMAGE_SHAPE)
                        except:
                            print(file, "cannot read, remove it.")
                            os.remove(file)
                            continue

                        if(to_image_type == '.png'):
                            cv2.imwrite(os.path.join(folder_1, filename+to_image_type), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
                        else:
                            cv2.imwrite(os.path.join(folder_1, filename+to_image_type), img)

                        os.remove(file)
                        #print("{}) {} transformed.".format(i, os.path.join(folder_1, filename+to_image_type)))

                i += 1
