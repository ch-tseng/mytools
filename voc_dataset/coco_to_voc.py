import json
import cv2
import imutils
import time
import os, glob

#http://cocodataset.org/#format-data

coco_annotations_path = "/DATA1/Datasets_download/Labeled/VOC/COCO/2014/annotations_train_valid/instances_train2014.json"
coco_images_path = "/DATA1/Datasets_download/Labeled/VOC/COCO/2014/train2014/"

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#output
datasetPath = "coco_2014/"
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

    for labelName, bbox_array in bboxes.items():
        for bbox in bbox_array:
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

    with open(coco_annotations_path) as json_file:
        data = json.load(json_file)

        info = data["info"]
        licenses = data["licenses"]
        images = data["images"]
        annotations = data["annotations"]
        classes = data["categories"]

        class_list = {}
        for class_data in classes:
            class_id = class_data['id']
            class_name = class_data['name']
            class_list.update( {class_id:class_name} )

        img_filename = {}
        for image_data in images:
            filename = image_data["file_name"]
            img_id = image_data["id"]
            img_filename.update( { filename:img_id } )

        img_bboxes = {}
        for image_data in annotations:
            img_id = image_data["image_id"]
            img_bbox_tmp = image_data["bbox"]

            img_bbox = []
            for num in img_bbox_tmp:
                img_bbox.append(int(num))

            if(img_id in img_bboxes):
                last_bbox = img_bboxes[img_id]
                last_bbox.append(img_bbox)
                img_bboxes.update( { img_id:last_bbox } )
            else:
                img_bboxes.update( { img_id:[img_bbox] } )

        img_class = {}
        for image_data in annotations:
            img_id = image_data["image_id"]
            category_id = image_data["category_id"]
            img_class.update( { img_id:class_list[category_id] } )

        print("Length:", len(class_list), len(img_filename), len(img_bboxes), len(img_class))
        for file in os.listdir(coco_images_path):
            file_name, file_extension = os.path.splitext(file)

            if(file in img_filename):
                bbox_objects = {}
                if(file in img_filename):
                    img_id = img_filename[file]
                    #print("img_id", img_id)
                    if(img_id in img_bboxes):
                        bboxes = img_bboxes[img_id]
                        #print("bboxes:", bboxes)

                        if(img_id in img_class):
                            classname = img_class[img_id]
                            #print("classname:", classname)
                            bbox_objects[img_class[img_id]] = bboxes

                            makeLabelFile(file_name, bbox_objects, os.path.join(coco_images_path, file))
