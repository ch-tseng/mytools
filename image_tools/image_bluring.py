import cv2
import imutils
import time
import os, glob
from tqdm import tqdm
import random
import numpy as np

image_path = "test_valid_images"
make_dataset_folder = "test_blur_images"
num_blur = 5
resize_to = (90, 45)

def averaging_blur(img, size=5):
    output = cv2.blur(img, (size, size))

    return output

def gaussian_blur(img, size=5):
    output = cv2.GaussianBlur(img, (size, size), cv2.BORDER_DEFAULT)

    return output

def median_blur(img, size=15):
    output = cv2.medianBlur(img, size)

    return output

def bilateral_blur(img, size=15, k1=60, k2=60):
    output = cv2.bilateralFilter(img, size, k1, k2)

    return output

def motion_blur1(img, size=15):
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)

    return output

def motion_blur2(img, size=15):
    # generating the kernel
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[:, int((size-1)/2)] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size

    # applying the kernel to the input image
    output = cv2.filter2D(img, -1, kernel_motion_blur)

    return output

def dynamic_bin(length):
    rtn = ""
    for i in range(0, length):
        a = random.randint(0,1)
        rtn = rtn + str(a)

    print(rtn)
    return rtn


if(not os.path.exists(make_dataset_folder)):
    os.makedirs(make_dataset_folder)

files = glob.glob(os.path.join(image_path, "*.jpg"))
files_bar = tqdm(files)


for file in files_bar:
    files_bar.set_description("Processing %s" % file)
    #type = random.randint(0,4)
    print(file)
    img = cv2.imread(file)
    img = cv2.resize(img, resize_to)

    

    for i in range(0, num_blur):
        id1 = random.randint(0,3)
        if(id1 == 0):
            strength = random.randint(1,3)
            if(strength % 2 == 0): strength += 1
            img = median_blur(img, size=strength)

        if(id1 == 1):
            strength = random.randint(20,60)
            if(strength % 2 == 0): strength += 1
            img = bilateral_blur(img, size=int(strength/3), k1=strength, k2=strength)

        #covert_types2 = dynamic_bin(5)
        #covert_types = "1000000"
        id2 = random.randint(0,4)
        #for id, type_c in enumerate(covert_types2):
        #    if(type_c == '1'):
        if(id2 == 0):
            strength = random.randint(7,9)
            if(strength % 2 == 0): strength += 1
            img = averaging_blur(img, size=strength)

        if(id2 == 1):
            strength = random.randint(7,9)
            if(strength % 2 == 0): strength += 1
            img = gaussian_blur(img, size=strength)

        if(id2 == 2):
            strength = random.randint(7,9)
            if(strength % 2 == 0): strength += 1
            img = motion_blur1(img, strength)

        if(id2 == 3):
            strength = random.randint(7,9)
            if(strength % 2 == 0): strength += 1
            img = motion_blur2(img, strength)

        if(id2 == 4):
            strength = random.randint(9,13)
            if(strength % 2 == 0): strength += 1
            img_c = motion_blur1(img, int(strength/2))
            img = motion_blur2(img_c, int(strength/2))


        base_name = os.path.basename(file)
        cv2.imwrite(os.path.join(make_dataset_folder, str(i) + '_' + str(id1) + '_' + str(id2) + '$' + base_name), img)