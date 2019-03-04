#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
import glob, os
from shutil import copyfile
import os.path
import cv2

#---------------------------------------------------------
NUM_GROUPS = 6
imageFolder = "/home/digits/datasets/USD_dollars/u1/images"
targetFolder = "/home/digits/datasets/USD_dollars/u1/split"

th_image_size = 1000  #image must larger than 1000 bytes (1KB)
unused_image_path = "/home/digits/datasets/palm_num/unused"  #move the images which can not use to here.
folderCharacter = "/"  # \\ is for windows
#--------------------------------------------------------

def chkEnv():
    if not os.path.exists(targetFolder):
        os.makedirs(targetFolder)

    if not os.path.exists(unused_image_path):
        os.makedirs(unused_image_path)

def filecount(folder):
    return len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])

#--------------------------------------------------------

chkEnv()

numTotalImages = 0
numImages = 0
numUnusedImages = 0
for file in os.listdir(imageFolder):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        file_size = os.path.getsize(imageFolder + folderCharacter + file)
        numTotalImages += 1

        if(file_size<th_image_size):  # image file must larger than 1KB
            os.rename(imageFolder + folderCharacter + file, unused_image_path + folderCharacter + file)
            numUnusedImages += 1
        else:
            numImages += 1

print("{}的相片總數共{}張".format(imageFolder, numTotalImages))
print("小於{}bytes無法使用的相片有{}張".format(th_image_size, numUnusedImages))
print("計有{}張相片可供label使用.".format(numImages))
print("-------------------------------------------------------------------------------")

num_split_images = int(numTotalImages / NUM_GROUPS)
num_extra_images = numTotalImages % NUM_GROUPS


num_group = 0
num_images = 0
index_file = 0
#spilt_num = num_split_images

print("check:", imageFolder)
for file in os.listdir(imageFolder):
    nameGroup = "G" + str(num_group+1)
    pathGroup = targetFolder + folderCharacter + nameGroup
    if(not os.path.exists(pathGroup)):
        os.makedirs(pathGroup)
    if(not os.path.exists(pathGroup + folderCharacter + "images")):
        os.makedirs(pathGroup + folderCharacter + "images")
    if(not os.path.exists(pathGroup + folderCharacter + "labels")):
        os.makedirs(pathGroup + folderCharacter + "labels")

    copyfile(imageFolder + folderCharacter + file, pathGroup + folderCharacter + "images" + folderCharacter + file)
    #os.rename(imageFolder + folderCharacter + file, pathGroup + folderCharacter + file)
    #num_images += 1
    #if(num_images>=spilt_num):
    num_group += 1
    index_file += 1
    print(index_file, num_group)

    if(num_group>=NUM_GROUPS):
        num_group = 0

print("共分為{}組".format(NUM_GROUPS))
for i in range(0, NUM_GROUPS):
    nameGroup = "G" + str(i+1)
    pathGroup = targetFolder + folderCharacter + nameGroup + folderCharacter + "images"
    print(pathGroup)
    print("第{}組 {}張".format(i+1, filecount(pathGroup)))
