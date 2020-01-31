import glob, os
import os.path
import shutil

#YOLO folder must has: *.jpg and *.txt
img_count_total = 40000  #more than real number is ok
source_image_type = ".jpg"
source_yololabel_type = ".txt"
file_count_in_folder = 1000
source_dataset = "/DATA1/Datasets_mine/labeled/crowd_human_dataset/yolo2"
target_dataset = "/DATA1/Datasets_mine/labeled/crowd_human_dataset/yolo_folders"

if not os.path.exists(target_dataset):
  os.makedirs(target_dataset)

for loop_folder in range(int(img_count_total/file_count_in_folder)+1):
  print("Loop count #{}".format(loop_folder))
  for i, file in enumerate(glob.iglob(os.path.join(source_dataset, "*"+source_image_type))):
    if(i>=file_count_in_folder):
      break

    filename = os.path.basename(file)
    file_mainname, file_extension = os.path.splitext(filename)

    source_img_file = os.path.join(source_dataset, filename )
    source_txt_file = os.path.join(source_dataset, file_mainname + source_yololabel_type )

    new_folder = os.path.join(target_dataset, str(loop_folder))
    if not os.path.exists( new_folder ):
      os.makedirs(new_folder)

    target_img_file = os.path.join(new_folder, filename )
    target_txt_file = os.path.join(new_folder, file_mainname + source_yololabel_type )
    
    try:
        print("#{}/{} move {},{}...".format(loop_folder, i, filename, file_mainname + source_yololabel_type))
        shutil.move(source_img_file, target_img_file)
        shutil.move(source_txt_file, target_txt_file)
    except:
        print("#{}/{} move filed".format(loop_folder, i))
        continue
