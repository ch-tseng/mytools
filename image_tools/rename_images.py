import os

dataset_images = r'D:\working\F'
to_path = r'D:\working\rename_F'
dataset_images = dataset_images.replace('\\', '/')
to_path = to_path.replace('\\', '/')
if not os.path.exists(to_path):
    os.makedirs(to_path)

i = 0
for file in os.listdir(dataset_images):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        i += 1
        print("Processing: ", os.path.join(dataset_images, file))
        new_filename = str(i).zfill(5) + file_extension
        os.rename(os.path.join(dataset_images, file), os.path.join(to_path, new_filename))