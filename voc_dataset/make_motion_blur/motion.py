# loading library 
import cv2 
import os
from libBlur import BLUR
import glob, shutil, random

ds_path = r"X:\Datasets\CH_custom\VOC\Human\gate_people_counter\dataset\V4_head_body\org_no_motion"
output_path = r"X:\Datasets\CH_custom\VOC\Human\gate_people_counter\dataset\V4_head_body\org_no_motion\motion_blur"

blurAUG = BLUR()
ds_path = ds_path.replace("\\", "/")
output_path = output_path.replace("\\", "/")
if not os.path.exists(output_path):
    os.makedirs(output_path)

image_ds_path = os.path.join(ds_path, "images")
xml_ds_path = os.path.join(ds_path, "labels")
if not os.path.exists( os.path.join(output_path, "images") ):
    os.makedirs(os.path.join(output_path, "images"))
if not os.path.exists(os.path.join(output_path, "labels")):
    os.makedirs(os.path.join(output_path, "labels"))

for id, file_path in enumerate(glob.glob(image_ds_path + "/*.jpg")):
    blurAUG.load_image(file_path)

    for i in range(1,5):
        strength = random.randint(30,120)
        img = blurAUG.motion_blur(blur_strength=strength, mtype=i)
        filename_img = "motion_" + str(strength) + "_{}_{}.jpg".format(id, i)
        filename_xml = "motion_" + str(strength) + "_{}_{}.xml".format(id, i)
        org_xml_path = os.path.join(xml_ds_path, os.path.basename(file_path).replace(".jpg", ".xml"))

        cv2.imwrite( os.path.join(output_path, "images", filename_img), img)
        shutil.copy(org_xml_path, os.path.join(output_path, "labels", filename_xml))


  
# Save the outputs. 
#cv2.imwrite('test.jpg', vertical_mb) 
#cv2.imwrite('test2.jpg', horizonal_mb) 