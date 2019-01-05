import cv2
from libMTCNN import faceMTCNN
detector = faceMTCNN()
min_face_size = 12

def getFaces(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = detector.detect_face(img, min_face_size)
    bboxes = []
    for face in faces:
        x = face[0]
        y = face[1]
        w = face[2]
        h = face[3]
        bboxes.append((x,y,w,h))

    return bboxes

pic = cv2.imread("peoples2.jpg")
faces = getFaces(pic)

for (x,y,w,h) in faces:
    cv2.rectangle( pic,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("TEST", pic)
cv2.waitKey(0)
