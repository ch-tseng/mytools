import random
import glob, os, sys
import os.path
from shutil import copyfile

#---------------------------------------------------------
testRatio = 0.2
validRatio = 0.2
dsFolder = "/home/digits/datasets/Diabnext_277_classes_org2/"
split_target_path = "/home/digits/datasets/Diabnext_split/"
valid_folder = "valid"
test_folder = "test"
train_folder = "train"
#--------------------------------------------------------

if not os.path.exists(dsFolder):
    print("No {} folder".format(dsFolder))
    sys.exit(1)

if not os.path.exists(split_target_path):
    os.makedirs(split_target_path)

valid_path = os.path.join(split_target_path, valid_folder)
test_path = os.path.join(split_target_path, test_folder)
train_path = os.path.join(split_target_path, train_folder)

if not os.path.exists(valid_path):
    os.makedirs(valid_path)

if not os.path.exists(test_path):
    os.makedirs(test_path)

if not os.path.exists(os.path.join(split_target_path, train_folder)):
    os.makedirs(train_path)


for folder_class in os.listdir(dsFolder):
    if(os.path.isdir(os.path.join(dsFolder,folder_class))):
        print("Folder:", folder_class)
        fileList = []

        for file in os.listdir(os.path.join(dsFolder,folder_class)):
            filePath = os.path.join(dsFolder,folder_class,file)
            if(os.path.isfile(filePath)):
                filename, file_extension = os.path.splitext(file)
                file_extension = file_extension.lower()

                if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
                    fileList.append(file)

        print("total image files for {} is:{}".format(folder_class, len(fileList)))

        a = range(len(fileList))
        test_valid_ratio = testRatio + validRatio
        if(test_valid_ratio>0):
            test_validCount = int(len(fileList) * test_valid_ratio)
            test_validData = random.sample(a, test_validCount)
            train_data = [x for x in a if x not in test_validData]

        if(testRatio>0):
            testCount = int(len(test_validData) * (testRatio/test_valid_ratio))
            test_data = random.sample(test_validData, testCount)
            valid_data = [x for x in test_validData if x not in test_data]

        #Copy file to valid folder
        for file_id in valid_data:
            file = fileList[file_id]
            org_path = os.path.join(dsFolder,folder_class,file )
            target_folder = os.path.join(split_target_path,valid_path,folder_class)
            target_path = os.path.join(split_target_path,valid_path,folder_class,file )
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            copyfile(org_path, target_path)
            print("copy {} to {}".format(file, target_folder))

        #Copy file to test folder
        for file_id in test_data:
            file = fileList[file_id]
            org_path = os.path.join(dsFolder,folder_class,file )
            target_folder = os.path.join(split_target_path,test_path,folder_class)
            target_path = os.path.join(split_target_path,test_path,folder_class,file )
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            copyfile(org_path, target_path)
            print("copy {} to {}".format(file, target_folder))

        #Copy file to train folder
        for file_id in train_data:
            file = fileList[file_id]
            org_path = os.path.join(dsFolder,folder_class,file )
            target_folder = os.path.join(split_target_path,train_path,folder_class)
            target_path = os.path.join(split_target_path,train_path,folder_class,file )
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            copyfile(org_path, target_path)
            print("copy {} to {}".format(file, target_folder))


        print ("Train:{} images".format(len(train_data)))
        print("Test:{} images".format(len(test_data)))
        print("Valid:{} images".format(len(valid_data)))

