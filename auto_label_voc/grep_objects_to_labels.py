from yoloOpencv import opencvYOLO
import cv2
import imutils
import time
import os

yolo = opencvYOLO(modeltype="yolov3", \
    objnames="../../../darknet/data/coco.names", \
    weights="../../../darknet/weights/yolov3.weights",\
    cfg="../../../darknet/cfg/yolov3.cfg")

objects = ("truck", "car", "motorcycle", "bus", "bicycle")
frame_interval =60

datasetPath = "vehicle_autolabel/"
imgPath = "images/"
labelPath = "labels/"
imgType = "jpg"  # jpg, png

inputType = "video"  # webcam, image, video
media = "/home/chtseng/works/alpr/videos/4K/IMG_2241.MOV"
write_video = True
video_out = "labeled.avi"
output_rotate = False
rotate = 0

xml_file = "xml_file.txt"
object_xml_file = "xml_object.txt"


def check_env():
    if not os.path.exists(os.path.join(datasetPath, imgPath)):
        os.makedirs(os.path.join(datasetPath, imgPath))

    if not os.path.exists(os.path.join(datasetPath, labelPath)):
        os.makedirs(os.path.join(datasetPath, labelPath))

    if not os.path.exists(os.path.join(datasetPath, labelPath)):
        os.makedirs(os.path.join(datasetPath, labelPath))


def chk_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

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

def makeLabelFile(img, bboxes):
    filename = str(time.time())
    jpgFilename = filename + "." + imgType
    xmlFilename = filename + ".xml"

    cv2.imwrite(os.path.join(datasetPath, imgPath, jpgFilename), img)

    xmlContent = generateXML(img, xmlFilename, os.path.join(datasetPath ,labelPath, xmlFilename), bboxes)
    file = open(os.path.join(datasetPath, labelPath, xmlFilename), "w")
    file.write(xmlContent)
    file.close


#fps count
start = time.time()
def fps_count(num_frames):
    end = time.time()
    seconds = end - start
    fps  = num_frames / seconds;
    print("    Estimated frames per second : {0}".format(fps))
    return fps

if __name__ == "__main__":
    check_env()

    if(inputType == "webcam"):
        INPUT = cv2.VideoCapture(0)
    elif(inputType == "image"):
        INPUT = cv2.imread(media)
    elif(inputType == "video"):
        INPUT = cv2.VideoCapture(media)

    if(inputType == "image"):
        yolo.getObject(INPUT, labelWant="", drawBox=True, bold=1, textsize=0.6, bcolor=(0,0,255), tcolor=(255,255,255))
        print ("Object counts:", yolo.objCounts)
        #print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
        #format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )
        #cv2.imshow("Frame", imutils.resize(INPUT, width=850))

        k = cv2.waitKey(0)
        if k == 0xFF & ord("q"):
            out.release()

    else:
        if(video_out!=""):
            width = int(INPUT.get(cv2.CAP_PROP_FRAME_WIDTH))   # float
            height = int(INPUT.get(cv2.CAP_PROP_FRAME_HEIGHT)) # float
            total_frames = int(INPUT.get(cv2.CAP_PROP_FRAME_COUNT))
            input("Total frames is "+str(total_frames)+", click ENTER to start.")

            if(write_video is True):
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                out = cv2.VideoWriter(os.path.join(datasetPath, "label_demo.avi"),fourcc, int(30.0/frame_interval), (int(width),int(height)))

        frameID = 0
        pic_id = 0
        obj_count = {}
        while True:
            hasFrame, frame = INPUT.read()
            if not hasFrame:
                print("Done processing !!!")
                break

            if(frameID % frame_interval == 0):

                if(output_rotate is True):
                    frame = imutils.rotate(frame, rotate)

                frame_org = frame.copy()

                yolo.getObject(frame, labelWant=objects, drawBox=True, bold=2, textsize=1.2, bcolor=(0,255,0), tcolor=(0,0,255))
                print("[FRAME #{}] image:{} labels:{}".format(total_frames-frameID, pic_id, yolo.labelNames))

                bbox_objects = {}

                for i, label in enumerate(yolo.labelNames):
                    #folder_path = os.path.join(output_path, label)
                    #chk_path(folder_path)
                    box = yolo.bbox[i]
                    x, y, w, h = box[0], box[1], box[2], box[3]
                    img_area = frame_org[y:y+h, x:x+w]

                    if(label in obj_count):
                        counts = obj_count[label]+1
                    else:
                        counts = 1

                    obj_count.update({label:counts})

                    if(label in bbox_objects):
                        bbox_objects[label].append([x,y,w,h])
                    else:
                        bbox_objects[label] = [[x,y,w,h]]

                    filename = label + '_' + str(counts) + '.jpg'
                    #print("save object file:", filename)
                    #cv2.imwrite(os.path.join(folder_path, filename), img_area)

                if(len(bbox_objects)>0):
                    makeLabelFile(frame_org, bbox_objects)

                #print("classIds:{}, confidences:{}, labelName:{}, bbox:{}".\
                #    format(len(yolo.classIds), len(yolo.scores), len(yolo.labelNames), len(yolo.bbox)) )

                #cv2.imshow("Frame", imutils.resize(frame, width=850))
                pic_id += 1
                #fps_count(frameID)

                if(write_video is True):
                    out.write(frame)

                k = cv2.waitKey(1)
                if k == 0xFF & ord("q"):
                    if(write_video is True):
                        out.release()

                    break

            frameID += 1
