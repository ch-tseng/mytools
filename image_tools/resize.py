import cv2
import os, glob
import imutils

imgFolder = "D:/works/official_car/plates"
new_imgFolder = "D:/works/official_car/plates2"
resize_type = 1  # 0:fixed size  1:based on long size
resize_0 = (300, 300)
resize_1 = 500

target_type = 0  #0: resize all imgs, 1:only resize the smaller imgs  2:only resize the larger imgs

if not os.path.exists(new_imgFolder):
	os.makedirs(new_imgFolder)

i = 0
for file in os.listdir(imgFolder):
    file_path = os.path.join(imgFolder, file)

    if(not os.path.isfile(file_path)):
        continue

    img = cv2.imread(file_path)

    file_error = False
    try:
        print("img shape:", img.shape)
        w, h = img.shape[1], img.shape[0]
    except:
        w, h = 0, 0
        file_error = True

    if(file_error is True):
        i += 1
        os.remove(file_path)
        print("  bad file, removed")
        continue

    resize = False
    if(target_type==0):
    	resize = True
    elif(target_type==1):
    	if(h<resize_1 and w<resize_1):
    		resize = True
    elif(target_type==2):
    	if(h>resize_1 or w>resize_1):
    		resize = True

    if(resize is True):    	

	    if(resize_type==0):
	    	img = cv2.resize(img, resize_0)
	    elif(resize_type==1):
	    	if(h>w):
	    		img = imutils.resize(img, height=resize_1)
	    	else:
	    		img = imutils.resize(img, width=resize_1)

    cv2.imwrite(os.path.join(new_imgFolder, file), img)    

        