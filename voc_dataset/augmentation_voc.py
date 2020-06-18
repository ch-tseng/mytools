import cv2
import imutils
import os, sys
import random
import numpy as np
from augvoc import augment

dataset_images = '/DATA1/Datasets_mine/labeled/fake_vegetables/images'
dataset_labels = '/DATA1/Datasets_mine/labeled/fake_vegetables/labels'

output_aug_images = '/DATA1/Datasets_mine/labeled/fake_vegetables/aug_images'
output_aug_labels = '/DATA1/Datasets_mine/labeled/fake_vegetables/aug_labels'

img_aug_count = 2
type_count = 2

if not os.path.exists(output_aug_images):
    os.makedirs(output_aug_images)
    print("no {} folder, created.".format(output_aug_images))

if not os.path.exists(output_aug_labels):
    os.makedirs(output_aug_labels)
    print("no {} folder, created.".format(output_aug_labels))

diverse_1 = {
    'rotate': [[-30, 30]],
    'shift': [[0.25]], #percent, shift up or down 0.25%
    'flip': True
}

diverse_2 = {
    'blur_avg': [[3, 9]], #kernel size
    'blur_gau': [[3, 9]], #kernel size
    'blur_med': [[3, 9]], #kernel size
    'blur_bil': [[3,9],[25,90],[25,90]], 
    'blur_mot': [[12,24]],
    'lighter': [[0.15, 0.5]],
    'darker': [[0.10, 0.30]],
    'noise': [[0.3,0.6]],
    'mosaic': [[5,35]],
    'imgs_mosaic': [[2,4]],
    'small2large': [[0.2, 0.5]],
    'contrast_more': [[-30, 60]],
    'contrast_less': [[0.2, 2.5]],
    'add_line': [[12]], #[max] border
    'add_square': [[30,120], [30,120]], #[[minW, minH], [maxW, maxH]]
    'add_circle': [[30,120]] #[[min, max]]
}

if __name__ == "__main__":
    augmentation = augment(dataset_images, dataset_labels, output_aug_images, output_aug_labels)

    augmentation.auto_make(img_aug_count, type_count)

    ''' Manual
    for id, file in enumerate(os.listdir(dataset_images)):
        #if(id>0):
        #    break

        filename, file_extension = os.path.splitext(file)
        file_extension = file_extension.lower()

        if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
            print("Processing: ", os.path.join(dataset_images, file))

            if not os.path.exists(os.path.join(dataset_labels, filename+".xml")):
                print("Cannot find the file {} for the image.".format(os.path.join(dataset_labels, filename+".xml")))

            else:
                image_path = os.path.join(dataset_images, file)
                xml_path = os.path.join(dataset_labels, filename+".xml")
                labelName, bboxes = augmentation.getLabels(image_path, xml_path)
                print("Grepped from XML: ", labelName, bboxes )

                img = cv2.imread(image_path)

                for count_num in range(0, img_aug_count):

                    ways = {0: 'no_change', 1:'shift', 2:'rotate', 3:'flip'}
                    con_id = random.randint(0, len(ways)-1)
                    if(con_id>0):
                        img, bboxes, labelName = augmentation.get_new_bbox(img, bboxes, labelName, ways[con_id])
                        print('way:', ways[con_id])

                    ways = random.sample([0, 1, 2, 3, 4, 5, 6, 7], 2)
                    ways_txt = ''

                    for way_id in ways:
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

                        ways_txt += str(way_id)

                    #plt.imshow(img) 
                    aug_filename = "{}_{}-{}-{}".format(filename, con_id, ways_txt, count_num)
                    cv2.imwrite(os.path.join(output_aug_images, aug_filename+file_extension), img)
                    for box in bboxes:
                        cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]), (255, 0, 0), 5)

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
                    print("write to -->", os.path.join(output_aug_labels, xmlFilename))
    '''
