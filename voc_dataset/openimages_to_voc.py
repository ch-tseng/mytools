import json
import time
import os, glob
import pathlib
from PIL import Image
import cv2
import sys
from tqdm import tqdm 

img_count_dataset = 50000  #split openimagesv6 to smaller dataset.

class_mapping = r'K:\OpenImage_v6\Annotations\class-descriptions-boxable.csv'
#want_class_list = { "Car", "Motorcycle", "Truck", "Bicycle", "Bus", "Van" }

want_class_list = []
#want_class_list = [ "Snake", "Crocodile", "Frog", "Turtle", "Tortoise", "Sea turtle", "Invertebrate", "Lizard", "Reptile", "Otter", "Jellyfish"]
annotations_path = r"K:\OpenImage_v6\Annotations\boxes\oidv6-train-annotations-bbox.csv"
images_path = r"K:\OpenImage_v6\dataset\train"

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#output
base_datasetPath = "E:/OpenImages_VOC/"
imgPath = "images/"
labelPath = "labels/"
imgType = "jpg"  # jpg, png

#----------------------------------------------------------------------------------------------------
class_mapping = class_mapping.replace('\\', '/')
annotations_path = annotations_path.replace('\\', '/')
images_path = images_path.replace('\\', '/')
datasetPath = base_datasetPath

class_maps = {}
revers_cmap = {}
target_class = []
class_mapping = 'class-descriptions-boxable.csv'
with open(class_mapping, 'r', encoding='UTF-8') as cfile:
    for line in cfile:
        line = line.replace('\n', '')
        cmap = line.split(',')
        class_maps.update( { cmap[1]:cmap[0]} )
        revers_cmap.update( { cmap[0]:cmap[1] })

no_class_list = ''
if want_class_list != []:
    for i, want_class in enumerate(want_class_list):
        if want_class not in class_maps:
            if i>0: no_class_list += ','
            no_class_list += want_class
        else:
            target_class.append(class_maps[want_class])

if no_class_list != '':
    print('No these class for OpenImage V6:', no_class_list)
    sys.exit()

def check_env():
    if not os.path.exists(os.path.join(datasetPath, imgPath)):
        os.makedirs(os.path.join(datasetPath, imgPath))

    if not os.path.exists(os.path.join(datasetPath, labelPath)):
        os.makedirs(os.path.join(datasetPath, labelPath))

def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[0] + bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[1] + bbox[3]))

    return file_updated

def generateXML(imgfile, filename, fullpath, bboxes, imgfilename):
    xmlObject = ""

    for (labelName, bbox) in bboxes:
        #for bbox in bbox_array:
        xmlObject = xmlObject + writeObjects(labelName, bbox)

    with open(xml_file) as file:
        xmlfile = file.read()

    img = cv2.imread(imgfile)
    #print(os.path.join(datasetPath, imgPath, imgfilename))
    cv2.imwrite(os.path.join(datasetPath, imgPath, imgfilename), img)

    (h, w, ch) = img.shape
    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", filename )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + filename )
    xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

    return xmlfile

def makeLabelFile(filename, bboxes, imgfile):
    jpgFilename = filename + "." + imgType
    xmlFilename = filename + ".xml"

    #cv2.imwrite(os.path.join(datasetPath, imgPath, jpgFilename), img)

    xmlContent = generateXML(imgfile, xmlFilename, os.path.join(datasetPath ,labelPath, xmlFilename), bboxes, jpgFilename)
    file = open(os.path.join(datasetPath, labelPath, xmlFilename), "w")
    file.write(xmlContent)
    file.close

file_list = {}
if __name__ == "__main__":
    #check_env()
    print('annotations_path', annotations_path)
    f = open(annotations_path)
    lines = f.readlines()
    total_lines = len(lines)
    lineID = 0
    for line in tqdm(lines):
        data_annotations = line.split(',')
        if(len(data_annotations)==21 and lineID>0):
            #print('data_annotations', data_annotations)
            class_id = data_annotations[2]
            if(class_id in target_class or target_class==[]):
                file_name = data_annotations[0]
                type_anno = data_annotations[2]

                minx,maxx,miny,maxy = data_annotations[4], data_annotations[5], data_annotations[6], data_annotations[7]
                #print('minx,maxx,miny,maxy ', minx,maxx,miny,maxy )
                if(file_name in file_list):
                    #print(file_list)
                    last_data = file_list[file_name]
                    #print("last_data 1:", last_data)
                    last_data.append((revers_cmap[type_anno], [minx,maxx,miny,maxy]))
                    #print("append:", (target_class[type_anno], [x,y,mx,my]))
                    #print("last_data 2:", last_data)
                    file_list.update( {file_name:last_data} )
                    #print("update file_list:", file_list)

                else:
                    file_list.update({ file_name:[(revers_cmap[type_anno], [minx,maxx,miny,maxy])]})

        lineID += 1

    id = 0
    count_now = 0
    ds_id = 0
    total_lines = len(file_list)
    for img_path in tqdm(pathlib.Path(images_path).iterdir()):
        if img_path.is_file():
            filename = os.path.basename(img_path)
            base_name, file_extension = os.path.splitext(filename)
            if(base_name in file_list):
                info = file_list[base_name]
                list_bboxes = []

                img_file = os.path.join(images_path, filename)
                im = Image.open(img_file)
                width, height = im.size

                for data in info:
                    classname = data[0]

                    minx, maxx, miny, maxy = float(data[1][0])*width, float(data[1][1])*width, float(data[1][2])*height, float(data[1][3])*height
                    xx,yy,ww,hh = int(minx), int(miny), int(maxx-minx), int(maxy-miny)
                    #print("[{}/{}] {}:{} ---> {},{}".format(id, total_lines, filename, data[1], classname, [xx,yy,ww,hh]))

                    list_bboxes.append((classname,[xx,yy,ww,hh]))

                folder_path_ds = "openimage_v6_voc_{}".format(ds_id)
                datasetPath = os.path.join(base_datasetPath, folder_path_ds)
                check_env()
                '''
                if not os.path.exists(datasetPath):
                    os.makedirs( os.path.join(datasetPath, 'images'))
                    os.makedirs( os.path.join(datasetPath, 'labels'))
                '''
                makeLabelFile(base_name, list_bboxes, img_file)
                count_now += 1

                if count_now>img_count_dataset:
                    count_now = 0
                    ds_id += 1

        id += 1