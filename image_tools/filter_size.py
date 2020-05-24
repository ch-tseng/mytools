import cv2
import os

img_path = "/Volumes/AIDATA1/dataset_Mine/mouth_mask/images"
min_size = 300*300

i = 0
for img_file in os.listdir(img_path):
    img_filepath = os.path.join(img_path, img_file)

    img = cv2.imread(img_filepath)
    if img is not None:
        if(img.shape[1]*img.shape[0]<min_size):
            os.remove(img_filepath)
            i += 1
            print("{}) size:{} removed, file: {}".format(i, img.shape[1]*img.shape[0], img_filepath)) 
    else:
        print(img_filepath, "is bad, remove it.")
        os.remove(img_filepath)
