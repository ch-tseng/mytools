import os
import shutil

ds_folder = "/home/digits/datasets/Diabnext_277_classes_org2"

for folder in os.listdir(ds_folder):
    if(os.path.isdir(os.path.join(ds_folder, folder))):
        split_folder = folder.split('.')
        if(len(split_folder)>1):
            name_folder = split_folder[1]
        else:
            name_folder = split_folder[0]

        split_folder2 = name_folder.split('(')
        name_folder2 = split_folder2[0]

        print(folder, '-->', name_folder2)
        shutil.move(os.path.join(ds_folder, folder), os.path.join(ds_folder, name_folder2))
