#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
import os.path
import shutil

ds_folder = "/home/digits/datasets/Diabnext_277_classes_org2"

for classname in os.listdir(ds_folder):
    folder_1  = os.path.join(ds_folder, classname)

    if(os.path.isdir(folder_1)):
        for subfolder in os.listdir(folder_1):
            folder_2  = os.path.join(folder_1, subfolder)

            if(os.path.isdir(folder_2)):
                for file in os.listdir(folder_2):
                    filepath = os.path.join(folder_2, file)
                    print("move file from {} to {}".format(filepath, os.path.join(folder_1, subfolder+'-'+file)))
                    shutil.move(filepath, os.path.join(folder_1, subfolder+'-'+file) )

                shutil.rmtree(folder_2)
