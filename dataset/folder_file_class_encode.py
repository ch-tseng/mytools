#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, time
import os.path
import shutil
import pandas as pd

#ds_folder = "/home/digits/datasets/Diabnext_org_dataset"
ds_folder = "/home/digits/datasets/Diabnext_277_classes_org2"
#keep these image types
image_types = ['.png']
#cfg_folder = 'cfg-diabnext-277'
cfg_folder = 'cfg-diabnext_300-277'
labels_list = 'labels.list'
CSV_FILE = "info.csv"

def id2code(idx):
    encode = idx.replace("0", "A")
    encode = encode.replace("1", "B")
    encode = encode.replace("2", "C")
    encode = encode.replace("3", "D")
    encode = encode.replace("4", "E")
    encode = encode.replace("5", "F")
    encode = encode.replace("6", "G")
    encode = encode.replace("7", "H")
    encode = encode.replace("8", "I")
    encode = encode.replace("9", "J")

    return encode

def class2id(classname):
    global root

    try:
        id = root.index(classname)

    except:
        id = None

    return id

def id2filename(classname, id):
    if(id<0):
        id = id * -1

    strID = str(int(id)).zfill(5)
    filename = strID + '_' + classname

    return filename


if not os.path.exists( cfg_folder ):
    print("folder created:", cfg_folder)
    os.makedirs(cfg_folder)

f = open( os.path.join(cfg_folder,labels_list) , "w")

root = os.listdir(ds_folder)
root = sorted(root)

IMAGES_NUM = 0
FOLDER_NUM = 0
INDEX = []
ENCODE = []
for classname in os.listdir(ds_folder):
    full_path = os.path.join(ds_folder, classname)

    strID = str(class2id(classname)).zfill(3)
    encode_name = id2code(strID)
    INDEX.append(strID)
    ENCODE.append(encode_name)

    #rename file name
    id = 0
    for file in os.listdir(full_path):
        filename, file_extension = os.path.splitext(file)
        img_path = os.path.join(full_path, file)
        img_filename = id2filename(encode_name, id) + file_extension

        print("    rename {} to {}".format(img_path, os.path.join(full_path, img_filename)))
        shutil.move(img_path, os.path.join(full_path, img_filename) )
        id += 1
        IMAGES_NUM += 1

    #rename folder name
    #print("rename {} to {}".format(full_path, os.path.join(ds_folder, encode_name)))
    #shutil.move(full_path, os.path.join(ds_folder, encode_name) )
    FOLDER_NUM += 1

dataframe = pd.DataFrame({'folder':root,
                          'images_number':IMAGES_NUM, 
                          'folder_number':FOLDER_NUM, 
                          'index':INDEX,
                          'encode':ENCODE,
                         })

dataframe.to_csv( os.path.join(cfg_folder,CSV_FILE), index=False, sep=',')

ENCODE.sort()
f.writelines("%s\n" % idencode for idencode in ENCODE)
f.close()
print("請注意{}檔案的label編號順序.".format(labels_list))
