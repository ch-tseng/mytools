# -*- coding:utf-8 -*-
# !/usr/bin/env python
 

import argparse
import json
import matplotlib.pyplot as plt
import skimage.io as io
import cv2
from labelme import utils
import numpy as np
import glob
import PIL.Image
 
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
 
class labelme2coco(object):
    def __init__(self, labelme_json=[], save_json_path='./tran.json'):

        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        # self.data_coco = {}
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
 
        self.save_json()
 
    def data_transfer(self):
 
        for num, json_file in enumerate(self.labelme_json):
            with open(json_file, 'r') as fp:
                data = json.load(fp) # load the json file
                self.images.append(self.image(data, num))
                for shapes in data['shapes']:
                    label = shapes['label']
                    if label not in self.label:
                        self.categories.append(self.categorie(label))
                        self.label.append(label)

                    points = shapes['points']# The point here is obtained by using the rectangle. There are only two points, which need to be converted into four points.
                    points.append([points[0][0],points[1][1]])
                    points.append([points[1][0],points[0][1]])
                    self.annotations.append(self.annotation(points, label, num))
                    self.annID += 1
 
    def image(self, data, num):
        image = {}
        img = utils.img_b64_to_arr(data['imageData']) # parsing the original image data
                 # img=io.imread(data['imagePath']) # Open the image via the image path
        # img = cv2.imread(data['imagePath'], 0)
        height, width = img.shape[:2]
        img = None
        image['height'] = height
        image['width'] = width
        image['id'] = num + 1
        image['file_name'] = data['imagePath'].split('/')[-1]
 
        self.height = height
        self.width = width
 
        return image
 
    def categorie(self, label):
        categorie = {}
        categorie['supercategory'] = 'Cancer'
        categorie['id'] = len(self.label) + 1 # 0 defaults to background
        categorie['name'] = label
        return categorie
 
    def annotation(self, points, label, num):
        annotation = {}
        annotation['segmentation'] = [list(np.asarray(points).flatten())]
        annotation['iscrowd'] = 0
        annotation['image_id'] = num + 1
                 # annotation['bbox'] = str(self.getbbox(points)) # When using the list to save the json file, I get an error (don't know why)
                 # list(map(int,a[1:-1].split(','))) a=annotation['bbox'] Use this method to convert to list
        annotation['bbox'] = list(map(float, self.getbbox(points)))
        annotation['area'] = annotation['bbox'][2] * annotation['bbox'][3]
        # annotation['category_id'] = self.getcatid(label)
        annotation['category_id'] = self.getcatid(label)#Note that the source code defaults to 1
        annotation['id'] = self.annID
        return annotation
 
    def getcatid(self, label):
        for categorie in self.categories:
            if label == categorie['name']:
                return categorie['id']
        return 1
 
    def getbbox(self, points):
        # img = np.zeros([self.height,self.width],np.uint8)
                 # cv2.polylines(img, [np.asarray(points)], True, 1, lineType=cv2.LINE_AA) # 
                 # cv2.fillPoly(img, [np.asarray(points)], 1) #  The internal pixel value is 1
        polygons = points
 
        mask = self.polygons_to_mask([self.height, self.width], polygons)
        return self.mask2box(mask)
 
    def mask2box(self, mask):

        # np.where(mask==1)
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        clos = index[:, 1]
                 # parsing the top left column number
        left_top_r = np.min(rows)  # y
        left_top_c = np.min(clos)  # x
 
                 # parsing the bottom right column number
        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(clos)
 
        # return [(left_top_r,left_top_c),(right_bottom_r,right_bottom_c)]
        # return [(left_top_c, left_top_r), (right_bottom_c, right_bottom_r)]
        # return [left_top_c, left_top_r, right_bottom_c, right_bottom_r]  # [x1,y1,x2,y2]
        return [left_top_c, left_top_r, right_bottom_c - left_top_c,right_bottom_r - left_top_r] # [x1,y1,w,h] Corresponds to the COCO bbox format
 
    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype=np.uint8)
        mask = PIL.Image.fromarray(mask)
        xy = list(map(tuple, polygons))
        PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
        mask = np.array(mask, dtype=bool)
        return mask
 
    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations
        return data_coco
 
    def save_json(self):
        self.data_transfer()
        self.data_coco = self.data2coco()
                 # save json file
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent=4, cls=MyEncoder) # indent=4 More beautiful display
 
 
labelme_json = glob.glob('J:/Practices/Punch_peoples/segs/*.json')
# labelme_json=['./Annotations/*.json']
 
labelme2coco(labelme_json, 'J:/Practices/Punch_peoples_coco_format/train.json')
