# -*- coding: utf-8 -*-
import os, time, glob

#save_to = "image2"
target_folder =  [ "Test-0/labels_D00", "Test-0/labels_D01", "Test-0/labels_D10", "Test-0/labels_D11", \
    "Test-0/labels_D20", "Test-0/labels_D21", "Test-0/labels_D40", "Test-0/labels_D42", "Test-0/labels_D91", "Test-0/images"]

#--------------------------------------------------------------------------------

def get_sorted_files(file_path):
    files = list(filter(os.path.isfile, glob.glob(file_path + "/*")))
    return files

    

if __name__ == '__main__':

    for id, folder in enumerate(target_folder):
        for id, file in enumerate(get_sorted_files(folder)):
            filename = os.path.basename(file)
            new_filename = filename.replace(' ','')
            print("    {} {} --> {}".format(id+1, file, new_filename))

            new_file_path = os.path.join(folder, new_filename)
            os.rename(file, new_file_path)