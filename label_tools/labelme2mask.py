import cv2
import json
import numpy as np
import os

classes_name = ['background', 'circle', 'square']
cls_map = {name: i for i, name in enumerate(classes_name)}
cls_map

img_path = 'H:/UNet_segmentation/datatset/road/images/'
json_path = 'H:/UNet_segmentation/datatset/road/seg/'
mask_save_path = 'H:/UNet_segmentation/datatset/road/masks'

if not os.path.exists(mask_save_path):
    os.makedirs(mask_save_path)

for file in os.listdir(json_path):
    
    filename, file_extension = os.path.splitext(file)

    if(file_extension.lower() != '.json'):
        continue

    json_file_path = os.path.join(json_path, file)
    data = json.load(open(json_file_path))
    height = data['imageHeight']
    width = data['imageWidth']
    imagePath = data['imagePath']
    img_filename = os.path.basename(imagePath)
    img_filepath = os.path.join(img_path, img_filename)
    img = cv2.imread(img_filepath)

    # Draw Object mask
    mask = np.zeros((len(classes_name), height, width))
    for shape in data['shapes']:
        cls_name = shape['label']
        cls_idx = cls_map[cls_name]
        points = shape['points']
        cv2.fillPoly(mask[cls_idx], np.array([points], dtype=np.int32), 255) # fill object with 255

    # update backgroud mask
    mask[0] = 255-np.max(mask[1:], axis=0)

    # show all masks
    #for m in mask:
    #    plt.imshow(m, cmap='gray')
    #    plt.show()

    # Get File name
    mask_filename = img_filename.split('.')[0]

    # Save Mask File
    for i, m in enumerate(mask):
        path_save = os.path.join(mask_save_path, f'{mask_filename}_mask_{i}.png')
        cv2.imwrite(path_save, m)