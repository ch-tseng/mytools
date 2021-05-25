import os
import glob
import random
import darknet
import time
import cv2
import numpy as np

'''
Usage:
from darkOpencv import darknetDetect

yolo = darknetDetect( config_file="models/yolov3_tiny/yolov3-tiny.cfg",
                      data_file="models/yolov3_tiny/obj.data",
                      weights="models/yolov3_tiny/yolov3-tiny_mask.weights" )

face_image, face_boxes, face_classes = yolo.detect( img.copy(), \
                #labelwant=['face_no_mask', 'face_other_covering', 'face_shield', 'face_with_mask', 'face_with_mask_incorrect'], \
                labelwant='',
                thresh=0.5)

'''

class darknetDetect:
    def __init__(self, config_file, data_file, weights):
        network, class_names, class_colors = darknet.load_network(
            config_file, data_file, weights, batch_size=1 )

        self.network = network
        self.class_names = class_names
        self.class_colors = class_colors

        width = darknet.network_width(network)
        height = darknet.network_height(network)
        self.width = width
        self.height = height

    def check_batch_shape(self, images, batch_size):
        """
            Image sizes should be the same width and height
        """
        shapes = [image.shape for image in images]
        if len(set(shapes)) > 1:
            raise ValueError("Images don't have same shape")
        if len(shapes) > batch_size:
            raise ValueError("Batch size higher than number of images")
        return shapes[0]

    def bbox2points(self, bbox):
        """
        From bounding box yolo format
        to corner points cv2 rectangle
        """
        x, y, w, h = bbox
        xmin = int(round(x - (w / 2)))
        xmax = int(round(x + (w / 2)))
        ymin = int(round(y - (h / 2)))
        ymax = int(round(y + (h / 2)))
        return xmin, ymin, xmax, ymax

    def image_detection(self, image, network, class_names, class_colors, thresh):
        # Darknet doesn't accept numpy images.
        # Create one with image we reuse for each detect
        width = self.width
        height = self.height
        darknet_image = darknet.make_image(width, height, 3)

        #image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                   interpolation=cv2.INTER_LINEAR)

        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
        darknet.free_image(darknet_image)
        image = darknet.draw_boxes(detections, image_resized, class_colors)

        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

    def image_classification(self, image, network, class_names):
        width = self.width
        height = self.height
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (width, height),
                                    interpolation=cv2.INTER_LINEAR)
        darknet_image = darknet.make_image(width, height, 3)
        darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = darknet.predict_image(network, darknet_image)
        predictions = [(name, detections[idx]) for idx, name in enumerate(class_names)]
        darknet.free_image(darknet_image)

        return sorted(predictions, key=lambda x: -x[1])

    def convert2relative(self, image, bbox):
        """
        YOLO format use relative coordinates for annotation
        """
        x, y, w, h = bbox
        height, width, _ = image.shape
        return int(x/width), int(y/height), int(w/width), int(h/height)

    def save_annotations(self, name, image, detections, class_names):
        """
        Files saved with image_name.txt and relative coordinates
        """
        file_name = os.path.splitext(name)[0] + ".txt"
        with open(file_name, "w") as f:
            for label, confidence, bbox in detections:
                x, y, w, h = self.convert2relative(image, bbox)
                print('convert', bbox, x,y,w,h)
                label = class_names.index(label)
                f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, x, y, w, h, float(confidence)))

    def detect(self, img, labelwant='', thresh=0.5):
        image, detections = self.image_detection(
            img, self.network, self.class_names, self.class_colors, thresh)

        bboxes, classes =[], []
        append = False
        if labelwant == '': append = True
        for label, confidence, bbox in detections:
            if labelwant !='' and len(labelwant)>0:
                if label in labelwant:
                    append = True

            if append is True:
                x1, y1, x2, y2 = self.bbox2points(bbox)
                #(x, y, w, h) = (x1, y1, x2-x1, y2-y1)
                ratio_x = img.shape[1]/self.width
                ratio_y = img.shape[0]/self.height
                (x, y, w, h) = (x1 * ratio_x, y1*ratio_y, (x2-x1)*ratio_x, (y2-y1)*ratio_y)
                #label = self.class_names.index(label)
                #bboxes.append( [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])] )
                bboxes.append( [int(x),int(y),int(w),int(h)] )
                classes.append(label)

        return image, bboxes, classes
