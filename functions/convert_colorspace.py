import cv2
import glob, os
from tqdm import tqdm 

img_path = "/DATA1/Datasets_mine/labeled/vehicles_coco_PASCAL_VOC/JPEGImages_rgb"
converted_to = "/DATA1/Datasets_mine/labeled/vehicles_coco_PASCAL_VOC/JPEGImages"
color_space = cv2.COLOR_RGB2HSV

if(not os.path.exists(converted_to)):
    os.makedirs(converted_to)

files = glob.glob(os.path.join(img_path, '*.jpg'))
#print(files)

pbar =  tqdm(files)
for file in pbar:
    pbar.set_description("Processing %s" % file) 
    img = cv2.imread(file)
    base_name = os.path.basename(file)
    filename, file_extension = os.path.splitext(file)

    img_converted = cv2.cvtColor(img, color_space)
    cv2.imwrite(os.path.join(converted_to, base_name), img_converted)
