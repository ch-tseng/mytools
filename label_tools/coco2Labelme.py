
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import imutils
import os, time
import os.path
import numpy as np
import json

#-------------------------------------------

new_labelme_path = '/DATA1/Datasets_mine/labeled/labelme_coco_vehicles'
coco_dataset_path = '/DATA1/Datasets_download/Labeled/VOC/COCO_Dataset/2017'
labels_want = [ 'truck', 'bus', 'car', 'motorcycle' ]
rename_lable = [ 'motorcycle', 'motorcycle', 'motorcycle', 'motorcycle' ]

if not os.path.exists(new_labelme_path):
    os.makedirs(new_labelme_path)
    print("no {} folder, created.".format(new_labelme_path))

if __name__ == '__main__':
    coco_anno_path = os.path.join(coco_dataset_path, 'annotations', 'instances_val2017.json')

    with open(coco_anno_path , 'r') as reader:
        jf = json.loads(reader.read())

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

        area = annot['area']
        annot_id = annot['id']

        #data = coco_images[img_id]
        if(img_id in coco_annot):
            seg_data = coco_annot[img_id][0]
            box_data = coco_annot[img_id][1]
        else:
            seg_data, box_data = [], []

        print("TEST:" , segmentations)
        seg_data.append(segmentations)
        box_data.append(box_data)

        coco_annot.update( {img_id:[seg_data, box_data]} )

    print(coco_annot)



