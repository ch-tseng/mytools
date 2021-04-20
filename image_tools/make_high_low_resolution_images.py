import cv2
from tqdm import tqdm
import os

high_res_ds = '/WORKS/Datasets/Faces/Asin_six_age_face/'
high_res_width = 384
low_res_width = 64

output_ds = 'res_ds'

#------------------------------------------------------------

output_h_path = os.path.join(output_ds, 'high')
output_l_path = os.path.join(output_ds, 'low')

if not os.path.exists(output_h_path):
    os.makedirs(output_h_path)
if not os.path.exists(output_l_path):
    os.makedirs(output_l_path)


i = 0
for file in tqdm(os.listdir(high_res_ds)):
    fname, fext = os.path.splitext(file)

    if fext.lower() in ['.jpg', '.png', '.jpeg']:
        file_path = os.path.join(high_res_ds, file)
        img = cv2.imread(file_path)

        img_h = cv2.resize(img, (high_res_width, high_res_width))
        img_l = cv2.resize(img, (low_res_width, low_res_width))
        filename = str(i).zfill(8) + '.jpg'
        cv2.imwrite( os.path.join(output_h_path, filename), img_h)
        cv2.imwrite( os.path.join(output_l_path, filename), img_l)

        i += 1


