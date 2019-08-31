
import scipy.io as sio 
from yoloOpencv import opencvYOLO
import cv2
import imutils
import time
import os, glob


oxford_hands_lbls_path = '/DATA1/Images/Labeled/VOC/hand_dataset/hand_dataset_oxford/hand_dataset/training_dataset/training_data/annotations/'
oxford_hands_imgs_path = '/DATA1/Images/Labeled/VOC/hand_dataset/hand_dataset_oxford/hand_dataset/training_dataset/training_data/images/'

label = "palm"
xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#output
datasetPath = "palm_oxford_hands/train"
imgPath = "images/"
labelPath = "labels/"
imgType = "jpg"  # jpg, png


def check_env():
    if not os.path.exists(os.path.join(datasetPath, imgPath)):
        os.makedirs(os.path.join(datasetPath, imgPath))

    if not os.path.exists(os.path.join(datasetPath, labelPath)):
        os.makedirs(os.path.join(datasetPath, labelPath))

    if not os.path.exists(os.path.join(datasetPath, imgPath)):
        os.makedirs(os.path.join(datasetPath, imgPath))


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    x, y, w, h = min(x_coordinates), min(y_coordinates), max(x_coordinates)-min(x_coordinates), max(y_coordinates)-min(y_coordinates)

    return [int(x), int(y), int(w), int(h)]

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
    print(os.path.join(datasetPath, imgPath, imgfilename))
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

def mat_box(mat_path):
    data_train = sio.loadmat(mat_path)
    #print(data_train)
    num = len(data_train['boxes'][0])

    bboxes = []
    for pt in data_train['boxes'][0]:
        #print(pt)
        pts = []
        for i, point in enumerate(pt):
            #print(point[0][0][0], point[0][1][0])
            pts.append( point[0][0][0])
            pts.append( point[0][1][0])
            pts.append( point[0][2][0])
            pts.append( point[0][3][0])

        bbox = bounding_box(pts)
        bboxes.append(bbox)

    #print("BBOX:", bboxes)
    return bboxes

#mat_box(os.path.join(oxford_hands_lbls_path, 'Inria_390.mat'))

if __name__ == "__main__":
    check_env()

    for file in os.listdir(oxford_hands_imgs_path):
        full_path = os.path.join(oxford_hands_imgs_path, file)

        bbox_objects = {}
        if(os.path.isfile(full_path)):
            filename, file_extension = os.path.splitext(file)
            if(file_extension.lower() in ('.jpg', '.png', '.jpeg')):
                #img = cv2.imread(full_path)
                file_basename = os.path.basename(filename)
                mat_filename = file_basename + '.mat'
                mat_filepath = os.path.join(oxford_hands_lbls_path, mat_filename)
                if(os.path.isfile(mat_filepath)):
                    bboxes = mat_box(mat_filepath)
                    for box in bboxes:

                        if(label in bbox_objects):
                            bbox_objects[label].append(box)
                        else:
                            bbox_objects[label] = [box]

                    makeLabelFile(file_basename, bbox_objects, full_path)

