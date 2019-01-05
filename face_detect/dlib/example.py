import cv2
import dlib

detector = dlib.get_frontal_face_detector()
dlib_detectorRatio = 1

def getFaces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector( gray , dlib_detectorRatio)
    bboxes = []
    for faceid, rect in enumerate(rects):
        (x, y, w, h) = rect_to_bb(rect)
        bboxes.append((x,y,w,h))

    return bboxes

pic = cv2.imread("peoples.jpg")
faces = getFaces(pic)

for (x,y,w,h) in faces:
    cv2.rectangle( pic,(x,y),(x+w,y+h),(0,255,0),2)

cv2.imshow("TEST", pic)
cv2.waitKey(0)

