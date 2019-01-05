#pip3 install mtcnn
#https://github.com/ipazc/mtcnn

import cv2
from mtcnn.mtcnn import MTCNN
detector = MTCNN()

def getFaces(img):
    faces = detector.detect_faces(img)
    bboxes = []
    for face in faces:
        x = face["box"][0]
        y = face["box"][1]
        w = face["box"][2]
        h = face["box"][3]

        bboxes.append((x,y,w,h))

    return bboxes

pic = cv2.imread("peoples2.jpg")
faces = getFaces(pic)

for (x,y,w,h) in faces:
    cv2.rectangle( pic,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("TEST", pic)
cv2.waitKey(0)

