import os
from random import shuffle
from math import floor
import cv2

split = 0.8

ds_folder = "/DATA1/Datasets_mine/labeled/vehicles_coco_PASCAL_VOC"
image_folder = "images"
label_folder = "labels"

imgs_path = os.path.join(ds_folder, "JPEGImages")
if os.path.isdir(os.path.join(ds_folder, image_folder)):
    os.rename(os.path.join(ds_folder, image_folder), os.path.join(ds_folder, "JPEGImages"))

if os.path.isdir(os.path.join(ds_folder, label_folder)):
    os.rename(os.path.join(ds_folder, label_folder), os.path.join(ds_folder, "Annotations"))

if not os.path.isdir(os.path.join(ds_folder, "ImageSets", "Main")):
    os.makedirs(os.path.join(ds_folder, "ImageSets", "Main"))

split_file1 = os.path.join(ds_folder, "ImageSets/Main/trainval.txt")
split_file2 = os.path.join(ds_folder, "ImageSets/Main/train.txt")
split_file3 = os.path.join(ds_folder, "ImageSets/Main/val.txt")

def get_file_list_from_dir(datadir):
    #all_files = os.listdir(os.path.abspath(datadir))
    #data_files = list(filter(lambda file: file.endswith('.data'), all_files))
    data_files = []
    f = open(split_file1, "w")

    for file in os.listdir(datadir):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension in ['.jpg', '.png', 'jpeg', 'bmp', 'pic']):
            if(os.path.isfile(os.path.join(ds_folder, "Annotations", filename+".xml")) ):
                try:
                    image = cv2.imread(os.path.join(datadir,file) )
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                except:
                    print("Read error:",os.path.join(datadir,file))
                    continue

                data_files.append(filename)
                f.write(filename + '\n')

            else:
                print("Not exists:", os.path.join(ds_folder, "Annotations", filename+".xml"))

    f.close()

    return data_files

def randomize_files(file_list):
    shuffle(file_list)
    return file_list

def get_training_and_testing_sets(file_list):
    split_index = floor(len(file_list) * split)
    training = file_list[:split_index]
    testing = file_list[split_index:]
    return training, testing


img_list = randomize_files(get_file_list_from_dir(imgs_path))

training, testing = get_training_and_testing_sets(img_list)

f = open(split_file2, "w")
for file in training:
    f.write(file + '\n')

f.close()

f = open(split_file3, "w")
for file in testing:
    f.write(file + '\n')

f.close()



