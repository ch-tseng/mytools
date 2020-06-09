# pip install labelme2coco
#Images(*.jpgs) and label files must in the same folder (in labelme_folder)

# import package
import labelme2coco

# set directory that contains labelme annotations and image files
labelme_folder = "K:/Datatsets/Mine/Test_CLoDSA/Labelme2/polygons"

# set path for coco json to be saved
save_json_path = "K:/Datatsets/Mine/Test_CLoDSA/Labelme2/instances_val.json"

# conert labelme annotations to coco
labelme2coco.convert(labelme_folder, save_json_path)