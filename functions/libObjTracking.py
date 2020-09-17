# How to use
#-----------------------------------------------
# from libObjTracking import OBJTRACK
# TRACK = OBJTRACK()
# TRACK.start(frameID=frameid, bboxes=yolo.bbox)  #first image
# tracking_info = TRACK.tracking(frameID=frameid, bboxes=yolo.bbox, th_iou=0.35, th_remove_ob=th_remove_ob)  #next image

import cv2
import numpy as np
from scipy import spatial
from PIL import ImageFont, ImageDraw, Image

class OBJTRACK:
    def __init__(self):
        self.th_iou = 0.9

    def printText(self, bg, txt, color=(0,255,0,0), size=0.7, pos=(0,0), type="Chinese"):
        (b,g,r,a) = color

        if(type=="English"):
            cv2.putText(bg,  txt, pos, cv2.FONT_HERSHEY_SIMPLEX, size,  (b,g,r), 2, cv2.LINE_AA)

        else:
            ## Use simsum.ttf to write Chinese.
            fontpath = "fonts/wt009.ttf"
            font = ImageFont.truetype(fontpath, int(size*10*4))
            img_pil = Image.fromarray(bg)
            draw = ImageDraw.Draw(img_pil)
            draw.text(pos,  txt, font = font, fill = (b, g, r, a))
            bg = np.array(img_pil)

        return bg

    def iou(self, boxA, boxB):  #(x,y,w,h)
        # determine the (x, y)-coordinates of the intersection rectangle
        if(boxA is None or boxB is None):
            return 0.0

        else:   
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xA2, yA2 = boxA[0] + boxA[2],  boxA[1] + boxA[3]
            xB2, yB2 = boxB[0] + boxB[2],  boxB[1] + boxB[3]

            xB = min(xA2, xB2)
            yB = min(yA2, yB2)
            interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]

            iou = interArea / float(boxAArea + boxBArea - interArea)
            
            return iou

    def start(self, frameID, bboxes):
        obj_info = {}

        for oid, box in enumerate(bboxes):
            obj_name = "{}_{}".format(str(frameID).zfill(6), oid)
            obj_info.update( {obj_name:[box, box, 0] }  )  #物件ID: [起啟bbox, 目前bbox, 消失幾個frame了]

        print("init:", obj_info)
        self.obj_info = obj_info

    def tracking(self, frameID, bboxes, th_iou, th_remove_ob):
        obj_info = self.obj_info
        self.th_iou = th_iou
        obj_info_new, min_idname, min_box = {}, None, None
        update_bboxes = []

        for obj_name in obj_info:
            [init_box, box2, count_lost] = obj_info[obj_name]
            count_lost += 1
            if(count_lost>=th_remove_ob):
                del obj_info[obj_name]

            obj_info.update( {obj_name: [init_box, box2, count_lost] } )

        for oid, box in enumerate(bboxes):
            min_dist, min_idname = 1.0, None

            for obj_name in obj_info:
                [init_box, box2, count_lost] = obj_info[obj_name]
                v_iou = self.iou(box, box2)

                dist = abs(1.0-abs(v_iou))
                if(min_dist>dist and dist<th_iou):
                    min_dist = dist
                    min_idname = obj_name
                    min_init_box = init_box

            if(min_idname is not None):
                obj_info.update( { min_idname:[min_init_box, box, 0] }  )
                update_bboxes.append(min_idname)
            else:
                obj_name = "{}_{}".format(str(frameID).zfill(6), oid)
                obj_info.update( { obj_name:[box, box, 0] }  )
                update_bboxes.append(obj_name)

        self.obj_info = obj_info

        return update_bboxes

        