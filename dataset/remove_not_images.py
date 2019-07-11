#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
import os.path
import shutil

ds_folder = "/home/digits/datasets/Diabnext_277_classes_org2"
#allowed image types
image_types = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']

for classname in os.listdir(ds_folder):
    folder_1  = os.path.join(ds_folder, classname)

    if(os.path.isdir(folder_1)):
        for subfolder in os.listdir(folder_1):
            file  = os.path.join(folder_1, subfolder)

            if(os.path.isdir(file)):
                shutil.rmtree(file)
            else:
                filename, file_extension = os.path.splitext(file)

                if( file_extension.lower() not in image_types):
                    print("remove file:", file)
                    os.remove(file)
