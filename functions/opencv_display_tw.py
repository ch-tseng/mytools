#!/home/pi/envs/plants/bin/python
# -*- coding: utf-8 -*-

from PIL import ImageFont, ImageDraw, Image

def printText(txt, bg, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
    (b,g,r,a) = color

    if(type=="English"):
        ## Use cv2.FONT_HERSHEY_XXX to write English.
        cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

    else:
        ## Use simsum.ttf to write Chinese.
        fontpath = "wt009.ttf"
        #print("TEST", txt)
        font = ImageFont.truetype(fontpath, int(size*10*3.2))
        img_pil = Image.fromarray(bg)
        draw = ImageDraw.Draw(img_pil)
        draw.text(pos,  txt, font = font, fill = (b, g, r, a))
        bg = np.array(img_pil)

    return bg
