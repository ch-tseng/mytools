import numpy as np

import cv2

from mss import mss

from PIL import Image

 

mon = {'left': 0, 'top': 200, 'width': 1280, 'height': 720}

 

with mss() as sct:

    screenShot = sct.grab(mon)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    out = cv2.VideoWriter("output.avi", fourcc, 12, (int(screenShot.width),int(screenShot.height)))

   

    while True:

        screenShot = sct.grab(mon)

        img = Image.frombytes(

            'RGB',

            (screenShot.width, screenShot.height),

            screenShot.rgb,

        )

       

        cv2_img = np.array(img)

        cv2_img = cv2_img[...,::-1]

        cv2.imshow('test', cv2_img)

        out.write(cv2_img)

        if cv2.waitKey(33) & 0xFF in (

            ord('q'),

            27,

        ):

            break

           

out.release()           

