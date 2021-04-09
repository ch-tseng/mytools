from xml.dom import minidom
from skimage import util
import cv2
import imutils
import os, sys
import random
import numpy as np
import glob

class augment():
    def __init__(self, dataset_images, dataset_labels, neg_images, output_img_path, output_xml_path, diverse_1=None, diverse_2=None):
        self.xml_file1 = "xml_file.txt"
        self.xml_file2 = "xml_object.txt"
        self.dataset_images = dataset_images
        self.dataset_labels = dataset_labels
        self.output_aug_images = output_img_path
        self.output_aug_labels = output_xml_path
        self.neg_images = neg_images

        if(diverse_1 is None):
            self.diverse_1 = {
                'rotate': [[-30, 30]],
                'shift': [[0.25]], #percent, shift up or down 0.25%
                'flip': True
            }

        else:
            self.diverse_1 = diverse_1

        if(diverse_2 is None):
            self.diverse_2 = {
                'blur_avg': [[3, 9]], #kernel size
                'blur_gau': [[3, 9]], #kernel size
                'blur_med': [[3, 9]], #kernel size
                'blur_bil': [[3,9],[25,90],[25,90]], 
                'blur_mot': [[12,24]],
                'lighter': [[0.15, 0.5]],
                'darker': [[0.10, 0.30]],
                'noise': [[0.2,0.5]],
                'mosaic': [[5,25]],
                'imgs_mosaic': [[2,4]],
                'small2large': [[0.2, 0.5]],
                'contrast_more': [[-30, 60]],
                'contrast_less': [[0.2, 2.5]],
                'add_line': [[8]], #[max] border
                'add_square': [[30,90], [30,90]], #[[minW, minH], [maxW, maxH]]
                'add_circle': [[30,90]] #[[min, max]]
            }

        else:
            self.diverse_2 = diverse_2

    def load_dataset(self, ds_path):
        ds_list = os.listdir(ds_path)
        self.ds_list = ds_list

    def load_negs(self, neg_path):
        neg_list = os.listdir(neg_path)
        self.neg_list = neg_list

    def load_augnegs(self, neg_path):
        neg_list = os.listdir(neg_path)
        self.augneg_list = neg_list

    def load_augdataset(self, ds_path):
        ds_list = os.listdir(ds_path)
        self.augds_list = ds_list


    def splice_4imgs(self, img):
        neg_list = self.augneg_list
        ds_list = self.augds_list

        h, w = img.shape[0], img.shape[1]

        canvas = np.zeros((h*2, w*2, 3), dtype = 'uint8')
        canvas[0:h, 0:w] = img

        img_files, ratios = [], []
        for i in range(0,3):
            img_file = ds_list[random.randint(0, len(ds_list)-1)]
            img_path = os.path.join(self.output_aug_images, img_file)

            img2 = cv2.imread(img_path)
            hh, ww = img2.shape[0], img2.shape[1]
            ratio = (w/ww, h/hh)
            ratios.append(ratio)

            img2 = cv2.resize(img2, (w,h))
            img_files.append(img_file)

            if i == 0:
                canvas[0:h, w:w+w] = img2
            if i == 1:
                canvas[h:h+h, 0:w] = img2
            if i == 2:
                canvas[h:h+h, w:w+w] = img2

        return canvas, img_files, ratios

    def overlay_neg(self, img, dir):
        neglist = self.neg_list

        alpha = random.randint(1,5) / 10
        w, h = img.shape[1], img.shape[0]
        neg_file = neglist[random.randint(0, len(neglist))]
        neg_img = cv2.imread( os.path.join(dir, neg_file) )
        neg_img = cv2.resize(neg_img, (w,h))

        image_new = cv2.addWeighted(neg_img, alpha, img, 1 - alpha, 0)

        return image_new

    def merge_img(self, frame1, frame2):
        h, w = frame1.shape[0], frame1.shape[1]
        hh, ww = frame2.shape[0], frame2.shape[1]
        h_ratio = h/hh
        w_ratio = w/ww

        alpha = 0.5
        frame2 = cv2.resize(frame2, (w,h))
        image_new = cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)

        return image_new, (w_ratio, h_ratio)

    def do_shift(self, img, mask, s_type, shift_value, shift_range):
        diverse_1 = self.diverse_1
        empty_img = np.zeros((img.shape[0], img.shape[1], 3), dtype = 'uint8')
        empty_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype = 'uint8')

        if(s_type==0):
            if(shift_value<0):
                empty_img[0:img.shape[0]+shift_value, :] = img[-shift_value:img.shape[0], :]
                empty_mask[0:mask.shape[0]+shift_value, :] = mask[-shift_value:mask.shape[0], :]
            else:
                empty_img[shift_value:img.shape[0], :] = img[0:img.shape[0]-shift_value, :]
                empty_mask[shift_value:mask.shape[0], :] = mask[0:mask.shape[0]-shift_value, :]

        if(s_type==1):
            if(shift_value<0):
                empty_img[:, 0:img.shape[1]+shift_value ] = img[:, -shift_value:img.shape[1]]
                empty_mask[:, 0:mask.shape[1]+shift_value ] = mask[:, -shift_value:mask.shape[1]]
            else:
                empty_img[:, shift_value:img.shape[1]] = img[:, 0:img.shape[1]-shift_value]
                empty_mask[:, shift_value:mask.shape[1]] = mask[:, 0:mask.shape[1]-shift_value]

        if(s_type==2):
            if(shift_value<0):
                empty_img[0:img.shape[0]+shift_value, 0:img.shape[1]+shift_value ] = img[-shift_value:img.shape[0]:, -shift_value:img.shape[1]]
                empty_mask[0:mask.shape[0]+shift_value, 0:mask.shape[1]+shift_value ] = mask[-shift_value:mask.shape[0], -shift_value:mask.shape[1]]
            else:
                empty_img[shift_value:img.shape[0], shift_value:img.shape[1]] = img[0:img.shape[0]-shift_value, 0:img.shape[1]-shift_value]
                empty_mask[shift_value:mask.shape[0], shift_value:mask.shape[1]] = mask[0:mask.shape[0]-shift_value, 0:mask.shape[1]-shift_value]


        return empty_img, empty_mask

    def draw_lines(self, img, count):
        diverse_2 = self.diverse_2
        overlay = img.copy()

        for i in range(0, count):
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            border = random.randint(1, diverse_2["add_line"][0][0])
            hv = random.randint(0,1)

            if(hv==0):
                point_left_x = random.randint(0, int(img.shape[1]/5)-1)
                point_left_y = random.randint(0, img.shape[0]-1)
                point_right_x = random.randint(int(4*img.shape[1]/5), img.shape[1]-1)
                point_right_y = random.randint(0, img.shape[0]-1)

            else:
                point_left_x = random.randint(0, img.shape[1]-1)
                point_left_y = random.randint(0, int(img.shape[0]/5)-1)
                point_right_x = random.randint(0, img.shape[1]-1)
                point_right_y = random.randint(int(4*img.shape[0]/5), img.shape[0]-1)

            alpha = random.randint(3,8) / 10
            
            cv2.line(overlay, (point_left_x, point_left_y), (point_right_x, point_right_y), color, border)
            image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        return image_new

    def draw_square(self, img, count):
        diverse_2 = self.diverse_2
        overlay = img.copy()
        for i in range(0, count):
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            width = random.randint(diverse_2["add_square"][0][0], diverse_2["add_square"][0][1] - 1)
            height = random.randint(diverse_2["add_square"][1][0], diverse_2["add_square"][1][1] - 1)
            border = random.randint(0, 5)

            if(border<3): border = -1

            point_left_x = random.randint(0, img.shape[1]-1)
            point_left_y = random.randint(0, img.shape[0]-1)

            alpha = random.randint(3,8) / 10
           
            cv2.rectangle(overlay, (point_left_x, point_left_y), (point_left_x+width, point_left_y+height), color, border)
            image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        return image_new

    def rgb2gray2rgb(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        return gray_3channel

    def draw_circle(self, img, count):
        diverse_2 = self.diverse_2
        overlay = img.copy()
        for i in range(0, count):
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            diameter = random.randint(10, diverse_2["add_circle"][0][0]-1)
            border = random.randint(1, 5)
            if(border<3): border = -1

            point_center_x = random.randint(0, img.shape[1]-diameter-1)
            point_center_y = random.randint(0, img.shape[0]-diameter-1)

            alpha = random.randint(3,8) / 10
            
            cv2.circle(overlay,(point_center_x, point_center_y), diameter, color, border)
            image_new = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

        return image_new

    def draw_circle_old(self, img, count):
        diverse_2 = self.diverse_2
        for i in range(0, count):
            color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
            diameter = random.randint(10, diverse_2["add_circle"][0][0]-1)
            border = random.randint(1, 5)
            if(border<3): border = -1

            point_center_x = random.randint(0, img.shape[1]-diameter-1)
            point_center_y = random.randint(0, img.shape[0]-diameter-1)

            img = cv2.circle(img,(point_center_x, point_center_y), diameter, color, border)

        return img

    def contrast_more(self, img):
        diverse_2 = self.diverse_2
        contrast = random.randint(diverse_2["contrast_more"][0][0], diverse_2["contrast_more"][0][1])
        brightness = random.randint(diverse_2["contrast_more"][0][0], diverse_2["contrast_more"][0][1])
        img = np.int16(img)
        img = img * (contrast/127+1) - contrast + brightness
        img = np.clip(img, 0, 255)
        img = np.uint8(img)

        return img

    def blur_img(self, img):
        b_type = random.randint(0,4)
        if(b_type==0):
            img = self.averaging_blur(img, size=random.randrange(3,11,2))
        elif(b_type==1):
            img = self.gaussian_blur(img, size=random.randrange(3,11,2))
        elif(b_type==2):
            img = self.median_blur(img, size=random.randrange(5,25,2))
        elif(b_type==3):
            img = self.bilateral_blur(img, size=random.randrange(5,25,2),  k1=random.randrange(30,90,2),  k2=random.randrange(30,90,2))
        elif(b_type==4):
            img = self.motion_blur(img, size=random.randrange(3,9,2))

        return img

    def averaging_blur(self, img, size=5):
        output = cv2.blur(img, (size, size))

        return output

    def gaussian_blur(self, img, size=5):
        output = cv2.GaussianBlur(img, (size, size), cv2.BORDER_DEFAULT)

        return output

    def median_blur(self, img, size=15):
        output = cv2.medianBlur(img, size)

        return output

    def bilateral_blur(self, img, size=15, k1=60, k2=60):
        output = cv2.bilateralFilter(img, size, k1, k2)

        return output

    def motion_blur(self, img, size=15):
        # generating the kernel
        kernel_motion_blur = np.zeros((size, size))
        kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
        kernel_motion_blur = kernel_motion_blur / size

        # applying the kernel to the input image
        output = cv2.filter2D(img, -1, kernel_motion_blur)

        return output

    def noisy(self, img):
        noise_typ = random.randint(0, 4)

        if noise_typ == 0: #gauss
            image = util.random_noise(img, mode='gaussian', clip=True)

        elif noise_typ == 1: #salt
            image = util.random_noise(img, mode='salt', amount=random.randrange(10,85, 15)/100, clip=True)

        elif noise_typ == 2: #pepper
            image = util.random_noise(img, mode='pepper', amount=random.randrange(10,85, 15)/100, clip=True)

        elif noise_typ == 3: #s&p
            image = util.random_noise(img, mode='s&p', amount=random.randrange(10,85, 15)/100, clip=True)

        elif noise_typ == 4: #speckle
            image = util.random_noise(img, mode='speckle', clip=True)

        image = image*255
        image = image.astype(np.uint8)

        return image   

    def do_mosaic(self, frame):
        diverse_2 = self.diverse_2
        x, y = 0, 0
        w, h = frame.shape[1], frame.shape[0]
        neighbor = random.randrange(diverse_2["mosaic"][0][0], diverse_2["mosaic"][0][1], 2)
        for i in range(0, h - neighbor, neighbor):  # 关键点0 减去neightbour 防止溢出
            for j in range(0, w - neighbor, neighbor):
                overlay = frame.copy()
                rect = [j + x, i + y, neighbor, neighbor]
                color = frame[i + y][j + x].tolist()  # 关键点1 tolist
                left_up = (rect[0], rect[1])
                right_down = (rect[0] + neighbor - 1, rect[1] + neighbor - 1)  # 关键点2 减去一个像素

                alpha = random.randint(2,6) / 10
                cv2.rectangle(overlay, left_up, right_down, color, -1)
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        return frame

    def do_small_larger(self, frame, s_ratio=0.4):
        h, w = frame.shape[0], frame.shape[1]
        smaller = cv2.resize(frame, (int(w*s_ratio), int(h*s_ratio)))
        larger = cv2.resize(smaller, (w, h))

        return larger

    def do_rotate(self, img, mask, angle):
        diverse_1 = self.diverse_1

        img = imutils.rotate_bound(img, angle)
        mask = imutils.rotate_bound(mask, angle)

        return img, mask

    def getLabels(self, imgFile, xmlFile):
        print(xmlFile)
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

        bboxes = []
        for i in range(0, len(labelName)):
            bboxes.append( [labelXmin[i], labelYmin[i], labelXmax[i]-labelXmin[i], labelYmax[i]-labelYmin[i]])

        return labelName, bboxes

    def do_imgchange(self, img, type_diverse):

        if(type_diverse[:6] == 'rotate'):
            angle = int(type_diverse[6:9])
            img = imutils.rotate_bound(img, angle)

        if(type_diverse == 'flip'):
            f_type = random.randint(-1,1)
            img = cv2.flip(img, f_type)

        return img

    def get_new_bbox(self, img, bboxes, labellist, type_diverse):
        diverse_1 = self.diverse_1               
        cimg = img.copy()
        h_img, w_img = img.shape[0], img.shape[1]
        boxes, labels = [], []
        mask_img = None

        if(type_diverse[:6] == 'rotate'):
            angle = int(type_diverse[6:9])
            #cimg, mask_img = self.do_rotate(img, mask_img, angle)
            
        if(type_diverse == 'shift'):
            shift_value = 0
            while shift_value == 0:
                s_type = random.randint(0,2)
                shift_range = int(diverse_1['shift'][0][0] * img.shape[0])
                shift_value = random.randint(-shift_range, shift_range)

            #cimg, mask_img = self.do_shift(img, mask_img, s_type, shift_value, shift_range)

        if(type_diverse == 'flip'):
            f_type = random.randint(-1,1)
            cimg = cv2.flip(img, f_type)
            #mask_img_org = cv2.flip(mask_img, f_type)

        for id, box in enumerate(bboxes):
            #create mask for the original image
            x,y,w,h = box[0], box[1], box[2], box[3]
            org_rect = np.zeros((h_img, w_img), dtype = 'uint8')
            mask_img = cv2.rectangle(org_rect, (x, y), (x+w, y+h), 255, -1)

            if(type_diverse[:6] == 'rotate'):
                cimg, mask_img = self.do_rotate(img, mask_img, angle)
                #cv2.imshow('test', imutils.resize(cimg, width=600))
                #cv2.imshow('test2', imutils.resize(mask_img, width=600))
                #cv2.waitKey(0)

            elif(type_diverse == 'shift'):
                cimg, mask_img = self.do_shift(img, mask_img, s_type, shift_value, shift_range)
                #cv2.imshow('test', imutils.resize(img, width=600))
                #cv2.imshow('test2', imutils.resize(mask_img, width=600))
                #cv2.waitKey(0)

            elif(type_diverse == 'flip'):
                mask_img = cv2.flip(mask_img, f_type)


            contours, hierarchy = cv2.findContours(mask_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            #for c in contours:
            if(len(contours)>0):
                (nx, ny, nw, nh) = cv2.boundingRect(contours[0])
                boxes.append([nx,ny, nw, nh])
                labels.append(labellist[id])
            #cv2.rectangle(frame, (left,top), (right,bottom), (255, 0, 0), 2)

        return cimg, boxes, labels

    def xmlLabels(self, imgFile, xmlFile):
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


    def writeObjects(self, label, bbox):
        with open(self.xml_file2) as file:
            file_content = file.read()

        file_updated = file_content.replace("{NAME}", label)
        file_updated = file_updated.replace("{XMIN}", str(bbox[0]))
        file_updated = file_updated.replace("{YMIN}", str(bbox[1]))
        file_updated = file_updated.replace("{XMAX}", str(bbox[2]))
        file_updated = file_updated.replace("{YMAX}", str(bbox[3]))
        #print("update:", file_updated)

        return file_updated

    def generateXML(self, img, file_name, fullpath, bboxes):
        xmlObject = ""

        (labelName, labelXmin, labelYmin, labelXmax, labelYmax) = bboxes
        for id in range(0, len(labelName)):
            xmlObject = xmlObject + self.writeObjects(labelName[id], (labelXmin[id], labelYmin[id], labelXmax[id], labelYmax[id]))
            #print(xmlObject)
            #print("----------------------------------------------------------------------")

        with open(self.xml_file1) as file:
            xmlfile = file.read()

        #print("before:", xmlfile)
        (h, w, ch) = img.shape
        xmlfile = xmlfile.replace( "{WIDTH}", str(w) )
        xmlfile = xmlfile.replace( "{HEIGHT}", str(h) )
        xmlfile = xmlfile.replace( "{FILENAME}", file_name )
        xmlfile = xmlfile.replace( "{PATH}", fullpath + file_name )
        xmlfile = xmlfile.replace( "{OBJECTS}", xmlObject )
        #print("replaced: ", xmlfile)

        return xmlfile

    def makeDatasetFile(self, img, img_filename, bboxes):
        file_name, file_ext = os.path.splitext(img_filename)
        jpgFilename = file_name + ".jpg"
        xmlFilename = file_name + ".xml"

        #cv2.imwrite(out_path + imgPath + jpgFilename, img)
        #print("write to -->", out_path + imgPath + jpgFilename)

        xmlContent = self.generateXML(img, xmlFilename, os.path.join(self.output_aug_labels, xmlFilename), bboxes)
        #file = open(os.path.join(self.output_aug_labels, xmlFilename), "w")
        #file.write(xmlContent)
        #file.close
        return xmlContent
        #print("write to -->", os.path.join(output_aug_labels, xmlFilename))

    def auto_make(self, img_aug_count, type_count):

        dataset_images = self.dataset_images
        dataset_labels = self.dataset_labels
        output_aug_images = self.output_aug_images
        output_aug_labels = self.output_aug_labels

        for id, file in enumerate(os.listdir(dataset_images)):
            #if(id>0):
            #    break

            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()

            if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
                #print("Processing: ", os.path.join(dataset_images, file))

                if os.path.exists(os.path.join(dataset_labels, filename+".xml")):
                    image_path = os.path.join(dataset_images, file)
                    xml_path = os.path.join(dataset_labels, filename+".xml")
                    labelName, bboxes = self.getLabels(image_path, xml_path)
                    #print("Grepped from XML: ", labelName, bboxes )

                    img_org = cv2.imread(image_path)

                    for count_num in range(0, img_aug_count):

                        ways = {0: 'no_change', 1:'shift', 2:'rotate', 3:'flip'}
                        con_id = random.randint(0, len(ways)-1)
                        if(con_id>0):
                            img, bboxes, labelName = self.get_new_bbox(img_org, bboxes, labelName, ways[con_id])
                            #print('way:', ways[con_id])

                        ways = random.sample([0, 1, 2, 3, 4, 5, 6, 7], type_count)
                        ways_txt = ''

                        for way_id in ways:
                            if(way_id == 1):
                                img = self.draw_lines(img,random.randint(10,50))
                            elif(way_id == 2):
                                img = self.draw_square(img, random.randint(10,50))
                            elif(way_id == 3):
                                img = self.draw_circle(img, random.randint(10,50))
                            elif(way_id == 4):
                                img = self.contrast_more(img)
                            elif(way_id == 5):
                                img = self.blur_img(img)
                            elif(way_id == 6):
                                img = self.noisy(img)
                            elif(way_id == 7):
                                img = self.do_mosaic(img)

                            ways_txt += str(way_id)

                        #plt.imshow(img)
                        aug_filename = "{}_{}-{}-{}".format(filename, con_id, ways_txt, count_num)
                        cv2.imwrite(os.path.join(output_aug_images, aug_filename+file_extension), img)
                        #for box in bboxes:
                        #    cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255, 0, 0), 5)

                        cv2.imwrite("output/{}".format(aug_filename+file_extension), img)

                        aug_labelName, aug_labelXmin, aug_labelYmin, aug_labelXmax, aug_labelYmax = [], [], [], [], []
                        for id, box in enumerate(bboxes):
                            aug_labelName.append(labelName[id])
                            aug_labelXmin.append(box[0])
                            aug_labelYmin.append(box[1])
                            aug_labelXmax.append(box[2]+box[0])
                            aug_labelYmax.append(box[3]+box[1])

                        #print("send:", aug_labelName, aug_labelXmin, aug_labelYmin, aug_labelXmax, aug_labelYmax)
                        xml_file = self.makeDatasetFile(img, file, (aug_labelName, aug_labelXmin, aug_labelYmin, aug_labelXmax, aug_labelYmax))
                        #print(xml_file)
                        xmlFilename = os.path.join(output_aug_labels, aug_filename + '.xml')
                        file_object = open(os.path.join(output_aug_labels, xmlFilename), "w")
                        file_object.write(xml_file)
                        file_object.close
                        #print("write to -->", os.path.join(output_aug_labels, xmlFilename))

        def mosaic_4imgs(self, file):
            #print(file)
            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()

            if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):

                if os.path.exists(os.path.join(output_aug_labels, filename+".xml")):
                    image_path = os.path.join(output_aug_images, file)
                    xml_path = os.path.join(output_aug_labels, filename+".xml")

                    img = cv2.imread(image_path)
                    org_h, org_w = img.shape[0], img.shape[1]

                    for count_num in tqdm(range(0, img_aug_count)):
                        labelName, bboxes = self.getLabels(image_path, xml_path)

                        aug_labelName, aug_labelXmin, aug_labelYmin, aug_labelXmax, aug_labelYmax = [], [], [], [], []
                        for id, box in enumerate(bboxes):
                            aug_labelName.append(labelName[id])
                            aug_labelXmin.append(box[0])
                            aug_labelYmin.append(box[1])
                            aug_labelXmax.append(box[2]+box[0])
                            aug_labelYmax.append(box[3]+box[1])

                        img, mfiles, ratios = self.splice_4imgs(img)
                        aug_filename = "{}_{}-{}-{}".format(filename, 'splice', '4imgs', count_num)
                        #print(os.path.join(output_aug_images, aug_filename+file_extension))
                        cv2.imwrite(os.path.join(output_aug_images, aug_filename+file_extension), img)
                        for i, sfile in enumerate(mfiles):
                            (w_ratio, h_ratio) = ratios[i]
                            mfile_name, mfile_extension = os.path.splitext(sfile)
                            mimg_path = os.path.join(output_aug_images, sfile)
                            mxml_path = os.path.join(output_aug_labels, mfile_name+'.xml')
                            mlabelName, mbboxes = self.getLabels( mimg_path, mxml_path)

                            for id, box in enumerate(mbboxes):
                                aug_labelName.append(mlabelName[id])
                                if i == 0:
                                    aug_labelXmin.append( org_w + int(box[0] * w_ratio))
                                    aug_labelYmin.append( int(box[1] * h_ratio))
                                    aug_labelXmax.append( org_w + int(int(box[2]+box[0]) * w_ratio))
                                    aug_labelYmax.append( int(int(box[3]+box[1]) * h_ratio))
                                if i == 1:
                                    aug_labelXmin.append( int(box[0] * w_ratio))
                                    aug_labelYmin.append( org_h + int(box[1] * h_ratio))
                                    aug_labelXmax.append( int(int(box[2]+box[0]) * w_ratio))
                                    aug_labelYmax.append( org_h + int(int(box[3]+box[1]) * h_ratio))
                                if i == 2:
                                    aug_labelXmin.append( org_w + int(box[0] * w_ratio))
                                    aug_labelYmin.append( org_h + int(box[1] * h_ratio))
                                    aug_labelXmax.append( org_w + int(int(box[2]+box[0]) * w_ratio))
                                    aug_labelYmax.append( org_h + int(int(box[3]+box[1]) * h_ratio))

                        xml_file = self.makeDatasetFile(img, file, (aug_labelName, aug_labelXmin, aug_labelYmin, aug_labelXmax, aug_labelYmax))
                        xmlFilename = os.path.join(output_aug_labels, aug_filename + '.xml')
                        file_object = open(os.path.join(output_aug_labels, xmlFilename), "w")
                        file_object.write(xml_file)
                        file_object.close
#------------------------------------------------------------------------------
