#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
import json
from tqdm import tqdm 
from shutil import copyfile

#-------------------------------------------

new_labelme_path = '/DATA1/Datasets_mine/labeled/labelme_coco_vehicles/seg'
new_labelme_img_path = '/DATA1/Datasets_mine/labeled/labelme_coco_vehicles/images'

coco_dataset_path = '/DATA1/Datasets_download/Labeled/VOC/COCO_Dataset/2017'
coco_images_path = '/DATA1/Datasets_download/Labeled/VOC/COCO_Dataset/2017/val2017'

labels_want = [ 'truck', 'bus', 'car', 'motorcycle' ]
rename_lable = [ 'motorcycle', 'motorcycle', 'motorcycle', 'motorcycle' ]

#-------------------------------------------

labelme_main = 'labelme_main.txt'
labelme_block = 'labelme_block.txt'

if not os.path.exists(new_labelme_path):
    os.makedirs(new_labelme_path)
    print("no {} folder, created.".format(new_labelme_path))

if not os.path.exists(new_labelme_img_path):
    os.makedirs(new_labelme_img_path)
    print("no {} folder, created.".format(new_labelme_img_path))


if __name__ == '__main__':

    #read data from json file 
    coco_anno_path = os.path.join(coco_dataset_path, 'annotations', 'instances_val2017.json')

    with open(coco_anno_path , 'r') as reader:
        jf = json.loads(reader.read())

    coco_catergories = {}
    cate_list = jf['categories']
    for cate_dict in cate_list:
        category_id = cate_dict['id']
        supercategory = cate_dict['supercategory']
        name = cate_dict['name']
        coco_catergories.update( {category_id:[supercategory, name]} )

    coco_images = {}
    img_list = jf['images']
    for img_info in img_list:
        img_filename = img_info['file_name']
        img_size = (int(img_info['width']), int(img_info['height']))
        img_id = img_info['id']
        #array for image_id, [ img filename, img size]
        coco_images.update({img_id:[img_filename, img_size, []]})

    coco_annot = {}
    annot_list = jf['annotations']
    for annot in annot_list:
        bbox = annot['bbox']
        img_id = annot['image_id']
        category_id = annot['category_id']
        iscrowd = annot['iscrowd']
        if(iscrowd == 1):
            segmentations = annot['segmentation']['counts']
        else:
            segmentations = annot['segmentation'][0]

        new_segmentations = []  # change [x,y,x,y,x,y,x,y...] to [[x,,y],[x,y]....]
        for i, point in enumerate(segmentations):
            if(i%2 == 1):
                new_segmentations.append([segmentations[i-1], point])

        area = annot['area']
        annot_id = annot['id']
        #data = coco_images[img_id]
        if(img_id in coco_annot):
            #print('coco_annot[img_id][0]:', coco_annot[img_id])
            seg_data = coco_annot[img_id]
            #print("seg_data:", seg_data)
            #box_data = [category_id, coco_annot[img_id][1]]
            #box_data.append(bbox)
        else:
            seg_data, box_data = [], []

        # (class id , seg points (x,y) )
        seg_data.append((category_id, new_segmentations))
        #box_data.append([category_id, bbox])

        coco_annot.update( {img_id:seg_data} )

    # finished read COCO json file
    
    f = open(labelme_block)
    seg_blocks = f.read()
    f.close()

    f = open(labelme_main)
    seg_main = f.read()
    f.close()


    blocks = []
    for id, image_id in enumerate(tqdm(coco_images)):
        #print(image_id)

        if(image_id in coco_annot):
            blocks = ""
            seg_datas = coco_annot[image_id] #[(class id, [seg poinsts]), (), () ... ]

            total_seg = len(seg_datas)
            for i, seg in enumerate(seg_datas):
                label = coco_catergories[seg[0]]
                points =  seg[1]
                #print(image_id, label, points)

                block = seg_blocks
                block = block.replace('<LABEL>', label[1])
                #print("points:", points)
                block = block.replace('<POINTS>', str(points))
                if(i<(total_seg-1)): block = block + ','

                blocks += '\n'+block

            img_file = coco_images[image_id][0]
            [img_basename, img_ext] = img_file.split('.')
            main = seg_main
            main = main.replace('<BLOCK_POLY>', blocks)
            main = main.replace('<IMG_PATH>', '../images/'+img_file)
            main = main.replace('<IMG_WIDTH>', coco_images[image_id][0][0])
            main = main.replace('<IMG_HEIGHT>', coco_images[image_id][0][1])

            copyfile(os.path.join(coco_images_path, img_file), os.path.join(new_labelme_img_path, img_file))
            #print(main)
            f = open(os.path.join(new_labelme_path,img_basename+'.json') , "w")
            f.write(main)
            f.close()
