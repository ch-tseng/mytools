import scipy.io as sio 
from yoloOpencv import opencvYOLO
import cv2
import imutils
import time
import os, glob


label_name = 'airplane'
lasot_1_class_path = '/WORK1/MyProjects/my_tracking_cnn/dataset/LaSOT/airplane/'
xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"

#output
datasetPath = "/WORK1/MyProjects/my_tracking_cnn/dataset/LaSOT_voc/airplane"
imgPath = "images/"
labelPath = "labels/"
imgType = "jpg"  # jpg, png

def check_env():
    if not os.path.exists(datasetPath):
        os.makedirs(os.path.join(datasetPath, imgPath))

    if not os.path.exists(os.path.join(datasetPath, imgPath)):
        os.makedirs(os.path.join(datasetPath, imgPath))

    if not os.path.exists(os.path.join(datasetPath, labelPath)):
        os.makedirs(os.path.join(datasetPath, labelPath))


def bounding_box(points):
    x_coordinates, y_coordinates = zip(*points)
    x, y, w, h = min(x_coordinates), min(y_coordinates), max(x_coordinates)-min(x_coordinates), max(y_coordinates)-min(y_coordinates)

    return [int(x), int(y), int(w), int(h)]

def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    print("TEST:", label, bbox)
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

def makeLabelFile(id, filename, bboxes, imgfile):
    jpgFilename = str(id)+ '_' + filename + "." + imgType
    xmlFilename = str(id)+ '_' + filename + ".xml"

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
        print(pt)
        for i, point in enumerate(pt):
            pts.append( [point[0][0][0][1], point[0][0][0][0]])
            pts.append( [point[0][1][0][1], point[0][1][0][0]])
            pts.append( [point[0][2][0][1], point[0][2][0][0]])
            pts.append( [point[0][3][0][1], point[0][3][0][0]])

        print(pts)
        bbox = bounding_box(pts)
        bboxes.append(bbox)

    print("BBOX:", bboxes)
    return bboxes

#mat_box(os.path.join(oxford_hands_lbls_path, 'Inria_390.mat'))

if __name__ == "__main__":
    check_env()

    for id, folder in enumerate(os.listdir(lasot_1_class_path)):
        folder_path = os.path.join(lasot_1_class_path, folder)
        if(os.path.isdir(folder_path)):
            img_folder = os.path.join(folder_path, 'img')
            label_path = os.path.join(folder_path, 'groundtruth.txt')
            f = open(label_path)
            lines = f.readlines()
            f.close()

            for file in os.listdir(img_folder):
                img_path = os.path.join(img_folder, file)
                bbox_objects = {}

                if(os.path.isfile(img_path)):
                    filename, file_extension = os.path.splitext(file)

                    if(file_extension.lower() in ('.jpg', '.png', '.jpeg')):
                        bbox_objects = {}
                        file_basename = os.path.basename(filename)

                        data = lines[int(file_basename)-1].replace('\n', '')
                        bbox = data.split(',')
                        bboxes = [[int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]]
                        print(file, file_basename, '-->', data)
                        bbox_objects[label_name] = bboxes

                        makeLabelFile(id, file_basename, bbox_objects, img_path)


