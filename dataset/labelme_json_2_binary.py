import cv2
import matplotlib.pyplot as plt
import json
import numpy as np
import os
from glob import glob
from tqdm.auto import tqdm

#參數設定 -----------------------------------------------------------------------------------------------
json_path = '/WORKS/working/dataset_road/segments/*.json'
classes_name = ['background', 'zenra' ]
output_masks_path = '/WORKS/working/dataset_road/masks'
#------------------------------------------------------------------------------------------------------------

cls_map = {name: i for i, name in enumerate(classes_name)}

if not os.path.exists(output_masks_path):
    print("create output path of ", output_masks_path)
    os.makedirs(output_masks_path)

def dilate_and_erode(mask_data, struc="ELLIPSE", size=(10, 10)):
    """
    膨胀侵蚀作用，得到粗略的trimap图
    :param mask_data: 读取的mask图数据
    :param struc: 结构方式
    :param size: 核大小
    :return:
    """
    if struc == "RECT":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, size)
    elif struc == "CORSS":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, size)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, size)

    msk = mask_data / 255

    dilated = cv2.dilate(msk, kernel, iterations=1) * 255
    eroded = cv2.erode(msk, kernel, iterations=1) * 255
    res = dilated.copy()
    res[((dilated == 255) & (eroded == 0))] = 128
    return res

def generate_trimap(mask_data,eroision_iter=6,dilate_iter=8):
    mask =  mask_data
    #mask = cv2.imread(mask,0)
    mask[mask==1] = 255
    d_kernel = np.ones((3,3))
    erode  = cv2.erode(mask,d_kernel,iterations=eroision_iter)
    dilate = cv2.dilate(mask,d_kernel,iterations=dilate_iter)
    unknown1 = cv2.bitwise_xor(erode,mask)
    unknown2 = cv2.bitwise_xor(dilate,mask)
    unknowns = cv2.add(unknown1,unknown2)
    unknowns[unknowns==255]=127
    trimap = cv2.add(mask,unknowns)
    # cv2.imwrite("mask.png",mask)
    # cv2.imwrite("dilate.png",dilate)
    # cv2.imwrite("tri.png",trimap)
    labels = trimap.copy()
    labels[trimap==127]=1 #unknown
    labels[trimap==255]=2 #foreground
    #cv2.imwrite(mask_path,labels)
    return labels

masks = []
files = glob(json_path)
for json_file in tqdm(files):
    print(json_file)
    data = json.load(open(json_file))
    height = data['imageHeight']
    width = data['imageWidth']
    mask = np.zeros((len(classes_name), height, width))
    for shape in data['shapes']:
        cls_name = shape['label']
        print(cls_name)

        cls_idx = cls_map[cls_name]
        points = shape['points']
        #用白色255填充object區域
        cv2.fillPoly(mask[cls_idx], np.array([points], dtype=np.int32), 255) 

    # 將非Object區域（即backgroud）用0填充
    mask[0] = 255-np.max(mask[1:], axis=0)
    masks.append(mask)

    #save mask file
    filename_img = os.path.split(json_file)[-1]
    filename = filename_img.split('.')[0]
    for i, m in enumerate(mask):
        cname = classes_name[i]
        path_save = os.path.join(output_masks_path, f'{filename}_mask_{cname}.png')
        cv2.imwrite(path_save, m)

        #save Trimap
        #tri = generate_trimap(m,eroision_iter=36,dilate_iter=64)
        tri = dilate_and_erode(m, struc="ELLIPSE", size=(10, 10))
        path_save = os.path.join(output_masks_path, f'{filename}_trimap_{cname}.png')
        cv2.imwrite(path_save, tri)
