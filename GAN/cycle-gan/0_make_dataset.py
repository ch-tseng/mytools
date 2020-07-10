import os
import cv2
# example of preparing the horses and zebra dataset
from os import listdir
from numpy import asarray
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed

path = '/DATA1/Datasets_download/GAN/lion2tiger/'
output_np_dataset = '/DATA1/Datasets_download/GAN/lion2tiger/lion2tiger_256.npz'
resize_to = (256,256)

# load all images in a directory into memory
def load_images(path, size=(256,256)):
    data_list = list()
    # enumerate filenames in directory, assume all are images
    for filename in listdir(path):
        # load and resize the image
        pixels = load_img(path + filename, target_size=size)
        # convert to numpy array
        pixels = img_to_array(pixels)
        # store
        data_list.append(pixels)

    return asarray(data_list)

# load dataset A
dataA1 = load_images(path + 'trainA/', size=resize_to)
dataAB = load_images(path + 'testA/', size=resize_to)
dataA = vstack((dataA1, dataAB))
print('Loaded dataA: ', dataA.shape)
# load dataset B
dataB1 = load_images(path + 'trainB/', size=resize_to)
dataB2 = load_images(path + 'testB/', size=resize_to)
dataB = vstack((dataB1, dataB2))
print('Loaded dataB: ', dataB.shape)
# save as compressed numpy array
savez_compressed(output_np_dataset, dataA, dataB)
print('Saved dataset: ', output_np_dataset)
