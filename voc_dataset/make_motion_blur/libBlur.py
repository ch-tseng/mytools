import cv2
import numpy as np
import random

class BLUR:
    def __init__(self):
        print("Blur class is called")

    def load_image(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError('Image not found')

    def mean_blur(self, ksize=(5, 5)):
        image = self.image
        # 使用均值模糊
        blurred = cv2.blur(image, ksize)
        self.blur = blurred

        return blurred
    
    def gaussian_blur(self, ksize=(5, 5), sigmaX=0):
        image = self.image
        # 使用高斯模糊
        blurred = cv2.GaussianBlur(image, ksize, sigmaX)
        self.blur = blurred
    
        return blurred
    
    def median_blur(self, ksize=5):
        image = self.image
        # 使用中值模糊
        blurred = cv2.medianBlur(image, ksize)
        self.blur = blurred

        return blurred
    
    def bilateral_blur(self, d=9, sigmaColor=75, sigmaSpace=75):
        image = self.image
        # 使用雙邊濾波
        blurred = cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)
        self.blur = blurred

        return blurred
    
    def motion_blur(self, blur_strength=90, mtype=0):  #0: random, 1: up, 2: down, 3: left, 4: right
        image = self.image
        # 自定义模糊强度
        blur_strength = random.randint(30, 120)

        if mtype == 0:
            direction = random.randint(1, 4)
        else:
            direction = mtype

        if direction == 1:
            # 上方向的运动模糊
            motion_blur_kernel_up = np.zeros((blur_strength, blur_strength))
            motion_blur_kernel_up[:, blur_strength // 2] = np.ones(blur_strength) / blur_strength
            blurred = cv2.filter2D(image, -1, motion_blur_kernel_up)

        elif direction == 2:
            # 下方向的运动模糊
            motion_blur_kernel_down = np.zeros((blur_strength, blur_strength))
            motion_blur_kernel_down[:, blur_strength // 2] = np.ones(blur_strength) / blur_strength
            motion_blur_kernel_down = np.flip(motion_blur_kernel_down, axis=0)
            blurred = cv2.filter2D(image, -1, motion_blur_kernel_down)

        elif direction == 3:
            # 左方向的运动模糊
            motion_blur_kernel_left = np.zeros((blur_strength, blur_strength))
            motion_blur_kernel_left[blur_strength // 2, :] = np.ones(blur_strength) / blur_strength
            blurred = cv2.filter2D(image, -1, motion_blur_kernel_left)

        elif direction == 4:
            # 右方向的运动模糊
            motion_blur_kernel_right = np.zeros((blur_strength, blur_strength))
            motion_blur_kernel_right[blur_strength // 2, :] = np.ones(blur_strength) / blur_strength
            motion_blur_kernel_right = np.flip(motion_blur_kernel_right, axis=1)
            blurred = cv2.filter2D(image, -1, motion_blur_kernel_right)


        return blurred
    
    def rotate_blur(self, angle=90):
        image = self.image
        # 定義旋轉模糊核心
        rows, cols = image.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        blurred = cv2.warpAffine(image, rotation_matrix, (cols, rows))

        self.blur = blurred

        return blurred

    def show_image(self, image, title='Image'):
        cv2.imshow('Original Image', self.image)
        cv2.imshow('Blurred Image', self.blur)
        cv2.waitKey(0)
        cv2.destroyAllWindows()