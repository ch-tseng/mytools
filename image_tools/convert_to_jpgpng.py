import os
import cv2

dataset_images = r'D:\working\F'
to_path = r'D:\working\C_F'
convert_type = '.jpg'
prefix = '202301_'

dataset_images = dataset_images.replace('\\', '/')
to_path = to_path.replace('\\', '/')
if not os.path.exists(to_path):
    os.makedirs(to_path)

i = 0
for file in os.listdir(dataset_images):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()
    file_path = os.path.join(dataset_images, file)
    
    
    try:
        img = cv2.imread(file_path)
        test = img.shape
    except:
        print("cannot read", file_path)
        continue

    i += 1
    print("Processing: ", os.path.join(dataset_images, file))
    new_filename = prefix + str(i).zfill(5) + convert_type
    to_file_path = os.path.join(to_path, new_filename)
    print(file_path, to_file_path)
    
    cv2.imwrite(to_file_path, img)