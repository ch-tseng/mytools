import json
import cv2
import imutils
import time
import os, glob

#http://cocodataset.org/#format-data

#target_class = ["car", "dog"]   #[] --> all
#target_class = ["car", "truck", "bus", "motorcycle", "bicycle"]
#coco_annotations_path = "/DATA1/Datasets_download/Labeled/VOC/COCO/2014/annotations_train_valid/instances_train2014.json"
ds_path = "/media/chtseng/MyDatasets/Dataset/Download/Dollar street trial/"

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#output
datasetPath = "/media/chtseng/MyDatasets/Dataset/Download/DollarStreet_VOC/"
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
    print("TEST:",bbox)
    if(bbox[0]<0): bbox[0]=0
    if(bbox[1]<0): bbox[1]=0
    if(bbox[2]<0): bbox[2]=0
    if(bbox[3]<0): bbox[3]=0

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

if __name__ == "__main__":
    check_env()

    for folder in os.listdir(ds_path):
        folder_path = os.path.join(ds_path, folder)
        if os.path.isdir(folder_path):
            imgs_path = os.path.join(folder_path, 'img')

            for path_img in os.listdir(imgs_path):
                file_name, file_extension = os.path.splitext(path_img)
                if file_extension in ['.jpg', '.jpeg', '.png']:
                    image_path = os.path.join(imgs_path, path_img)
                    json_path = os.path.join(folder_path, 'ann', path_img+'.json')
                    if os.path.isfile(json_path):
                        f = open(json_path)
                        data = json.load(f)
                        objects = data['objects']

                        img_bboxes = []
                        class_list = {}
                        if len(objects)>0:
                            for ob in objects:
                                img_id = ob["id"]
                                category_id = ob["classId"]
                                class_name = ob['classTitle']
                                if ob['geometryType'] == 'rectangle':
                                    xy1 = ob['points']['exterior'][0]
                                    xy2 = ob['points']['exterior'][1]
                                    x, y, w, h = xy1[0], xy1[1], xy2[0]-xy1[0], xy2[1]-xy1[1]
                                    img_bboxes.append((class_name, [x,y,w,h]))

                        if len(img_bboxes)>0:
                            makeLabelFile(file_name, img_bboxes, image_path)
                        else:
                            print(data)
