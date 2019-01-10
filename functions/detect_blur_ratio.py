#Blur detection with OpenCV
#https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
import cv2
import os
import imutils

def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

imgPath = "/media/sf_datasets/blur/"

for img_file in os.listdir(imgPath):
    img = cv2.imread(imgPath + "/" + img_file)
    #img = imutils.resize(img, width=160)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurRate = variance_of_laplacian(gray)
    text = "clear:"
    cv2.putText(img, "{}: {:.2f}".format(text, blurRate), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("IM", img)
    cv2.imwrite("/media/sf_datasets/blur_output/" + img_file, img)
    cv2.waitKey(0)
