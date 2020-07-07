import cv2, os

img_path = "C:/Users/ch.tseng/Downloads/crowd"
width = 90
output_folder = "C:/Users/ch.tseng/Downloads/cuts"

if(not os.path.exists(output_folder)):
    os.makedirs(output_folder)


for i, file in enumerate(os.listdir(img_path)):
    filename, file_extension = os.path.splitext(file)
    file_extension = file_extension.lower()

    if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
        img = cv2.imread(os.path.join(img_path, file))
        ii = 0
        for r in range(0,img.shape[0],width):
            for c in range(0,img.shape[1],width):
                path_save = os.path.join(output_folder, filename + '_' + str(ii) + '.jpg')
                cv2.imwrite(path_save,img[r:r+width, c:c+width,:])
                ii+=1
