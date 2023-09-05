from configparser import ConfigParser
import ast
import glob, os
import cv2
from xml.dom import minidom

DS_path = "/DS/Datasets/CH_custom/VOC/Others/Device_numbers/dataset3/"
xml_path = "/DS/Datasets/CH_custom/VOC/Others/Device_numbers/dataset2/labels/"
output_dataset = "/DS/Datasets/CH_custom/VOC/Others/Device_numbers/dataset4/"


if not os.path.exists(output_dataset):
    os.makedirs(output_dataset)
if not os.path.exists( os.path.join(output_dataset, "images")):
    os.makedirs(os.path.join(output_dataset, "images"))
if not os.path.exists(os.path.join(output_dataset, "labels")):
    os.makedirs(os.path.join(output_dataset, "labels"))

def getLabels(xmlFile):
    labelXML = minidom.parse(xmlFile)
    labelName = []
    labelXmin = []
    labelYmin = []
    labelXmax = []
    labelYmax = []
    totalW = 0
    totalH = 0
    countLabels = 0

    tmpArrays = labelXML.getElementsByTagName("name")
    for elem in tmpArrays:
        labelName.append(str(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmin")
    for elem in tmpArrays:
        labelXmin.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymin")
    for elem in tmpArrays:
        labelYmin.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("xmax")
    for elem in tmpArrays:
        labelXmax.append(int(elem.firstChild.data))

    tmpArrays = labelXML.getElementsByTagName("ymax")
    for elem in tmpArrays:
        labelYmax.append(int(elem.firstChild.data))

    
    return labelName, labelXmin, labelYmin, labelXmax, labelYmax

def writeObjects(label, bbox):
    with open(object_xml_file) as file:
        file_content = file.read()

    file_updated = file_content.replace("{NAME}", label)
    file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
    file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
    file_updated = file_updated.replace("{XMAX}", str(bbox[0] + bbox[2]))
    file_updated = file_updated.replace("{YMAX}", str(bbox[1] + bbox[3]))

    return file_updated

def generateXML(img, filename, fullpath, bboxes):
    xmlObject = ""

    for labelName, bbox_array in bboxes.items():
        for bbox in bbox_array:
            xmlObject = xmlObject + writeObjects(labelName, bbox)

    with open(xml_file) as file:
        xmlfile = file.read()

    (h, w, ch) = img.shape
    xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
    xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
    xmlfile = xmlfile.replace( "{FILENAME}", filename )
    xmlfile = xmlfile.replace( "{PATH}", fullpath + filename )
    xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )

    return xmlfile

def makeLabelFile(img, bboxes, file_name):
    img_path = os.path.join(output_dataset, 'images')
    #file_name, ext_name = os.path.splitext(img_file_name)
    jpgFilename = file_name + ".jpg"
    xmlFilename = file_name + ".xml"

    #org_xml_file = os.path.join(xml_path,xmlFilename )
    output_xml_file = os.path.join(output_dataset, 'labels', xmlFilename)    
    #print('    org_xml_file', org_xml_file)
    print('    output_xml_file', output_xml_file)

    xmlContent = generateXML(img, xmlFilename, output_xml_file, bboxes)
    file = open(output_xml_file, "w")
    file.write(xmlContent)
    file.close    

    print(os.path.join(output_dataset, 'images', jpgFilename))
    cv2.imwrite(os.path.join(output_dataset, 'images', jpgFilename) , img)

cfg = ConfigParser()
cfg.read("crop_labeled_config.ini",encoding="utf-8")

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"
obj_count = {}
for dname in ast.literal_eval(cfg.get("dataset", "img_folders")):
    cropareas = ast.literal_eval(cfg.get(dname, "cropareas"))

    dpath = os.path.join(DS_path, dname)
    for img_path in glob.glob( os.path.join(dpath, "*.jpg") ):
        basname = os.path.basename(img_path).replace('.jpg','')
        xml_filepath = os.path.join( xml_path, basname+'.xml')
        if os.path.exists(xml_filepath):
            labelName, labelXmin, labelYmin, labelXmax, labelYmax = getLabels(xml_filepath)

            for aid, area in enumerate(cropareas):
                bbox_objects = {}


                for i, lname in enumerate(labelName):
                    (x,y,w,h) = (labelXmin[i],labelYmin[i],labelXmax[i]-labelXmin[i],labelYmax[i]-labelYmin[i])
                    cx,cy = (labelXmax[i]+labelXmin[i])/2, (labelYmax[i]+labelYmin[i])/2

                    if cx>area[0] and cx<(area[0]+area[2]) and cy>area[1] and cy<(area[1]+area[3]):
                        xx = x - area[0]
                        yy = y - area[1]
                        if xx<0: xx = 0
                        if yy<0: yy = 0

                        if lname in obj_count:
                            c = obj_count[lname]
                        else:
                            c = 0

                        obj_count.update({lname:c+1})


                        if lname in bbox_objects:
                            bbox_objects[lname].append([xx,yy,w,h])
                        else:
                            bbox_objects[lname] = [[xx,yy,w,h]]

                if(len(bbox_objects)>0):
                    img = cv2.imread(img_path)
                    cropped = img[area[1]:area[1]+area[3], area[0]:area[0]+area[2]]
                    makeLabelFile(cropped, bbox_objects, basname+'_'+str(aid))

print(obj_count)
