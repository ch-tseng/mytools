import json
import time
import os, glob
import pathlib
from PIL import Image
import cv2

'''
/m/0k4j,Car
/m/04_sv,Motorcycle
/m/07r04,Truck
/m/0199g,Bicycle
/m/01bjv,Bus
/m/0h2r6,Van


/m/01bl7v,Boy
/m/05r655,Girl

/m/0dzct,Human face
/m/02p0tk3,Human body
/m/03bt1vf,Woman
/m/04hgtk,Human head
/m/04yx4,Man
'''

target_class = { "/m/0k4j":"Car", "/m/04_sv":"Motorcycle", "/m/07r04":"Truck", "/m/0199g":"Bicycle", "/m/01bjv":"Bus", "/m/0h2r6":"Van" }
annotations_path = "/mnt/Open DataSet/train-annotations-bbox.csv"
images_path = "/mnt/Open DataSet/train_00/"

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#output
datasetPath = "OpenImages_v5_Vehicles/"
imgPath = "images/"
labelPath = "labels/"
imgType = "jpg"  # jpg, png

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
    check_env()

    f = open(annotations_path)
    lines = f.readlines()
    total_lines = len(lines)
    for lineID, line in enumerate(lines):
        data_annotations = line.split(',')
        if(len(data_annotations)==13):
            class_id = data_annotations[2]
            if(class_id in target_class):
                file_name = data_annotations[0]
                type_anno = data_annotations[2]

                minx,maxx,miny,maxy = data_annotations[4], data_annotations[5], data_annotations[6], data_annotations[7]

                if(file_name in file_list):
                    #print(file_list)
                    last_data = file_list[file_name]
                    #print("last_data 1:", last_data)
                    last_data.append((target_class[type_anno], [minx,maxx,miny,maxy]))
                    #print("append:", (target_class[type_anno], [x,y,mx,my]))
                    #print("last_data 2:", last_data)
                    file_list.update( {file_name:last_data} )
                    #print("update file_list:", file_list)

                else:
                    file_list.update({ file_name:[(target_class[type_anno], [minx,maxx,miny,maxy])]})


    total_lines = len(file_list)
    for id, img_path in enumerate(pathlib.Path(images_path).iterdir()):
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
                    print("[{}/{}] {}:{} ---> {},{}".format(id, total_lines, filename, data[1], classname, [xx,yy,ww,hh]))

                    list_bboxes.append((classname,[xx,yy,ww,hh]))

                makeLabelFile(base_name, list_bboxes, img_file)
