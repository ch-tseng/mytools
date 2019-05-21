#!/usr/bin/env python3
# -*- coding: utf-8 -*-

##################################################################
# move images in sub-sub-folders up to sub-folder (class folder) #
# remove not images and hidden files                             #
# correct .jpeg to .jpg                                          #

import cv2
import imutils
import os, time
import os.path
import shutil

target_ds = "/home/digits/datasets/Diabnext_French_Food"

file_type = [".jpg", ".jpeg", ".bmp", "gif", ".png"]

#-------------------------------------------------------------

if not os.path.exists(target_ds):
    sys.exit("No such path:", target_ds)

def process_file(file_correct, file_fullpath):

    if(os.path.isfile(file_fullpath)):
        filename_path, file_extension = os.path.splitext(file_fullpath)
        file_name_no_ext = os.path.basename(filename_path)

        file_extension_lower = file_extension.lower()

        if(file_extension_lower!=file_extension):
            print("rename file from {} to {}".format(file_name_no_ext+file_extension, file_name_no_ext+file_extension_lower))
            shutil.move( file_fullpath, filename_path+file_extension_lower)

        if(file_extension_lower in file_type):
            ext_name = str(time.time())[-6:]

            if(file_extension_lower == ".jpeg"):
                print("rename file from {} to {}".format(filename_path+file_extension_lower, filename_path+".jpg"))
                #shutil.move(filename_path+file_extension_lower, filename_path+".jpg" )
                shutil.move(filename_path+file_extension_lower, os.path.join(file_correct, file_name_no_ext+ext_name+".jpg"))
            else:
                shutil.move(filename_path+file_extension_lower, os.path.join(file_correct, file_name_no_ext+ext_name+file_extension))

        else:
            print("remove the file which is not image: ", filename_path+file_extension_lower)
            os.remove(filename_path+file_extension_lower)


def move_images(target_path):
    if(os.path.isdir(target_path)):

        for file in os.listdir(target_path):
            full_filepath = os.path.join(target_path, file)

            if(os.path.isfile(full_filepath)):
                check_lower_file_ext(full_filepath)


                filename, file_extension = os.path.splitext(file)
                file_extension = file_extension.lower()

              

                #remove the hidden files
                if(filename.startswith('.')):
                    os.remove(os.path.join(source_path,file))

                else:
                    if(file_extension in file_type):
                        if file_extension=='.jpeg':
                            file_extension='.jpg'

                        shutil.move(os.path.join(target_path,file), os.path.join(target_path, filename+file_extension) )

                        if(file_extension != '.jpg'):
                            print("warning! not jpg file: {}".format(os.path.join(target_path, filename+file_extension)))

                    else:
                        print("remove the file not image type: ", filename+file_extension)
                        os.remove(os.path.join(source_path,file))


def preprocess(correct_path, target_path):

    if(os.path.isdir(target_path)):
        print("processing sub-folder: {}".format(target_path))

        for file in os.listdir(target_path):
            process_file(correct_path, os.path.join(target_path, file))

    else:
        filename_path, file_extension = os.path.splitext(target_path)
        file_name_no_ext = os.path.basename(filename_path)

        file_extension_lower = file_extension.lower()

        if(file_extension_lower not in file_type):
            print("remove {}".format(target_path))
            os.remove(target_path)

#-------------------------------------------------------------

for classname in os.listdir(target_ds):
    full_path = os.path.join(target_ds, classname)

    #folder or file?
    if(os.path.isdir(full_path)):
        print("processing {} folder".format(classname))

        for file in os.listdir(full_path):
            full_path2 = os.path.join(full_path, file)

            if(os.path.isdir(full_path)):
                preprocess(full_path, full_path2)

    else:
        print("remove {} file".format(classname))
        os.remove(full_path)
