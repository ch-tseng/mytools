
import scipy.io as sio 
from yoloOpencv import opencvYOLO
import cv2
import imutils
import time
import os, glob


egohands_path = '/WORK1/dataset/egohands'

label = "palm"
xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#output
datasetPath = "palm_egohands/"
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

    print(imgfile)
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

data_train = sio.loadmat(os.path.join('/WORK1/dataset/egohands/metadata.mat'))
#data_train = sio.loadmat(os.path.join('/WORK1/dataset/egohands/PUZZLE_OFFICE_T_S/polygons.mat'))
print( data_train["video"][0][3][1][0])

if __name__ == "__main__":
    check_env()

    folder_count = len(data_train["video"][0])
    for id_folder in range(folder_count):
        folder_name = data_train["video"][0][id_folder][0][0]
        frame_count =  len(data_train["video"][0][id_folder][6][0])
        print(folder_name)
        for id_frame in range(0, frame_count):
            bbox_objects = {}
            frame_number = str(data_train["video"][0][id_folder][6][0][id_frame][0][0][0])
            frame_basename = "frame_"+frame_number.zfill(4)
            frame_filename = frame_basename + ".jpg"
            folder_path = os.path.join(egohands_path, folder_name)
            frame_filepath = os.path.join(egohands_path, folder_name, frame_filename)
            print("    Frame name:", frame_filepath)
            print("    label count:", len(data_train["video"][0][id_folder][6][0][id_frame]))
            print("    label count:", data_train["video"][0][id_folder][6][0][id_frame][0])
            print("    label count:", len(data_train["video"][0][id_folder][6][0][id_frame][1]))
            print("    label count:", len(data_train["video"][0][id_folder][6][0][id_frame][2]))
            print("    label count:", len(data_train["video"][0][id_folder][6][0][id_frame][3]))
            print("    label count:", len(data_train["video"][0][id_folder][6][0][id_frame][4]))

            
            bbox_objects = {}
            for i in range(1,5):
                counts = len(data_train["video"][0][id_folder][6][0][id_frame][i])
                if(counts>0):
                    points = data_train["video"][0][id_folder][6][0][id_frame][i]

                    bbox = bounding_box(points)
                    if(label in bbox_objects):
                        bbox_objects[label].append(bbox)
                    else:
                        bbox_objects[label] = [bbox]


                    print(bbox)

            makeLabelFile(frame_basename, bbox_objects, frame_filepath)
