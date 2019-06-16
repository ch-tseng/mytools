#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
import os.path
import shutil

ds_folder = "/home/digits/datasets/Diabnext_277_classes_org2"
#keep these image types
image_types = ['.png']
cfg_folder = 'cfg-diabnext_300-277'
ds_imglist = 'filelist.list'

if not os.path.exists( cfg_folder ):
    print("folder created:", cfg_folder)
    os.makedirs(cfg_folder)

f = open( os.path.join(cfg_folder,ds_imglist) , "w")

for classname in os.listdir(ds_folder):
    folder_1  = os.path.join(ds_folder, classname)

    if(os.path.isdir(folder_1)):
        i = 0
        for file in os.listdir(folder_1):
            filename, file_extension = os.path.splitext(file)

            if(file_extension.lower() in image_types):
                imgfile  = os.path.join(folder_1, file)
                f.write(imgfile+'\n')

print(os.path.join(cfg_folder,ds_imglist), "created.")
f.close()
