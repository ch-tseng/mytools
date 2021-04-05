import cv2
import imutils
import os, sys
import random
import numpy as np
from augvoc import augment
from tqdm import tqdm

dataset_images = r'D:\temp\total_crowdhuman\images'
dataset_labels = r'D:\temp\total_crowdhuman\labels'
neg_images = r'D:\temp\total_crowdhuman\negatives'

output_aug_images = r'D:\temp\total_crowdhuman\aug_images'
output_aug_labels = r'D:\temp\total_crowdhuman\aug_labels'
output_aug_negs = r'D:\temp\total_crowdhuman\negatives'

img_aug_count = 1

dataset_images = dataset_images.replace('\\', '/')
dataset_labels = dataset_labels.replace('\\', '/')
output_aug_images = output_aug_images.replace('\\', '/')
output_aug_labels = output_aug_labels.replace('\\', '/')
output_aug_negs = output_aug_negs.replace('\\', '/')
neg_images = neg_images.replace('\\', '/')

if not os.path.exists(output_aug_images):
    os.makedirs(output_aug_images)
    print("no {} folder, created.".format(output_aug_images))

if not os.path.exists(output_aug_labels):
    os.makedirs(output_aug_labels)
    print("no {} folder, created.".format(output_aug_labels))

if not os.path.exists(output_aug_negs):
    os.makedirs(output_aug_negs)
    print("no {} folder, created.".format(output_aug_negs))

diverse_1 = {
    'rotate': [[-10, 10], [80,100], [170,190], [260,280]],
    'shift': [[0.15]], #percent, shift up or down 0.25%
    'flip': False
}

diverse_2 = {
    'blur_avg': [[3, 5]], #kernel size
    'blur_gau': [[3, 5]], #kernel size
    'blur_med': [[3, 5]], #kernel size
    'blur_bil': [[3,7],[25,60],[25,60]], 
    'blur_mot': [[12,24]],
    'lighter': [[0.15, 0.5]],
    'darker': [[0.10, 0.30]],
    'noise': [[0.2,0.5]],
    'mosaic': [[5,5]],
    'imgs_mosaic': [[2,2]],
    'small2large': [[0.2, 0.5]],
    'contrast_more': [[-30, 60]],
    'contrast_less': [[0.2, 2.5]],
    'add_line': [[15]], #[max] border
    'add_square': [[5,5], [60, 50]], #[[minW, minH], [maxW, maxH]]
    'add_circle': [[15,60]] #[[min, max]]
}

if __name__ == "__main__":
    augmentation = augment(dataset_images, dataset_labels, output_aug_images, output_aug_labels)
    
    #augmentation.auto_make(img_aug_count, type_count)
    print("Generate negatives images from ", neg_images)
    #negatives
    for id, file in tqdm(enumerate(os.listdir(neg_images))):
        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            image_path = os.path.join(neg_images, file)
            #print("Processing: ", image_path)

            for count_num in range(0, img_aug_count):
                    
                    #ways = {0: 'no_change', 1:'rotate90', 2:'rotate180', 3:'rotate270', 4:'flip', 5:'shift' }
                    ways = {0: 'no_change', 1:'rotate90', 2:'rotate180', 3:'rotate270' }
                    for con_id in ways:
                        cimg = cv2.imread(image_path)
                        try:
                            test = cimg.shape
                        except:
                            break

                        cimg = augmentation.do_imgchange(cimg, ways[con_id])
                        #for way_id in [ 0, 1, 2, 3, 4, 5, 6]:
                        for way_id in [ 0, 1, 2, 3, 4, 5, 6]:
                            img = cimg.copy()
                            if(way_id == 1):
                                img = augmentation.draw_lines(img,random.randint(20,50))
                            elif(way_id == 2):
                                img = augmentation.draw_square(img, random.randint(20,50))
                            elif(way_id == 3):
                                img = augmentation.draw_circle(img, random.randint(20,50))
                            elif(way_id == 4):
                                img = augmentation.contrast_more(img)
                            elif(way_id == 5):
                                img = augmentation.blur_img(img)
                            elif(way_id == 6):
                                img = augmentation.noisy(img)
                            elif(way_id == 7):
                                img = augmentation.do_mosaic(img)

                            ways_txt = str(way_id)

                            #plt.imshow(img) 
                            aug_filename = "{}_{}-{}-{}".format(filename, con_id, ways_txt, count_num)
                            cv2.imwrite(os.path.join(output_aug_negs, aug_filename+file_extension), img)

    
    print("Generate training images from ", output_aug_negs)
    augmentation.load_negs(output_aug_negs)
    #Manual
    for id, file in tqdm(enumerate(os.listdir(dataset_images))):
        #if(id>0):
        #    break

        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            #print("Processing: ", os.path.join(dataset_images, file))

            if os.path.exists(os.path.join(dataset_labels, filename+".xml")):
                image_path = os.path.join(dataset_images, file)
                xml_path = os.path.join(dataset_labels, filename+".xml")
                

                for count_num in range(0, img_aug_count):
                    
                    #ways = {0: 'no_change', 1:'rotate90', 2:'rotate180', 3:'rotate270', 4:'flip', 5:'shift' }
                    ways = {0: 'no_change', 1:'rotate90', 2:'rotate180', 3:'rotate270' }
                    for con_id in ways:
                        img_org = cv2.imread(image_path)
                        try:
                            test = img_org.shape
                        except:
                            break

                        labelName, bboxes = augmentation.getLabels(image_path, xml_path)
                        #print("Grepped from XML: ", labelName, bboxes )

                        cimg, bboxes, labelName = augmentation.get_new_bbox(img_org, bboxes, labelName, ways[con_id])
                        #print('way:', ways[con_id]) 
                        
                        #for way_id in [ 0, 1, 2, 3, 4, 5, 6, 8]:
                        for way_id in [ 0, 4, 5, 6 ]:
                            img = cimg.copy()
                            if(way_id == 1):
                                img = augmentation.draw_lines(img,random.randint(5,25))
                            elif(way_id == 2):
                                img = augmentation.draw_square(img, random.randint(5,25))
                            elif(way_id == 3):
                                img = augmentation.draw_circle(img, random.randint(5,25))
                            elif(way_id == 4):
                                img = augmentation.contrast_more(img)
                            elif(way_id == 5):
                                img = augmentation.blur_img(img)
                            elif(way_id == 6):
                                img = augmentation.noisy(img)
                            elif(way_id == 7):
                                img = augmentation.do_mosaic(img)
                            elif(way_id == 8):
                                img = augmentation.overlay_neg(img, output_aug_negs)

                            ways_txt = str(way_id)

                            #plt.imshow(img) 
                            aug_filename = "{}_{}-{}-{}".format(filename, con_id, ways_txt, count_num)
                            cv2.imwrite(os.path.join(output_aug_images, aug_filename+file_extension), img)
                            #for box in bboxes:
                            #    cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255, 0, 0), 5)

                            cv2.imwrite("output/{}".format(aug_filename+file_extension), img)
                            #cv2.imwrite("output2.jpg", mask)

                            #print("TEST:", output_aug_images, file)
                            #image_path = os.path.join(output_aug_images, file)
                            #xml_path = os.path.join(output_aug_labels, filename+".xml")

                            aug_labelName, aug_labelXmin, aug_labelYmin, aug_labelXmax, aug_labelYmax = [], [], [], [], []
                            for id, box in enumerate(bboxes):
                                aug_labelName.append(labelName[id])
                                aug_labelXmin.append(box[0])
                                aug_labelYmin.append(box[1])
                                aug_labelXmax.append(box[2]+box[0])
                                aug_labelYmax.append(box[3]+box[1])

                            #print("send:", aug_labelName, aug_labelXmin, aug_labelYmin, aug_labelXmax, aug_labelYmax)
                            xml_file = augmentation.makeDatasetFile(img, file, (aug_labelName, aug_labelXmin, aug_labelYmin, aug_labelXmax, aug_labelYmax))
                            #print(xml_file)
                            xmlFilename = os.path.join(output_aug_labels, aug_filename + '.xml')
                            file_object = open(os.path.join(output_aug_labels, xmlFilename), "w")
                            file_object.write(xml_file)
                            file_object.close
                            #print("write to -->", os.path.join(output_aug_labels, xmlFilename))

