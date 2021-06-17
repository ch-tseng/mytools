import cv2
import imutils
import os, sys
import random
import numpy as np
from augvoc import augment
from tqdm import tqdm

dataset_base = r'/WORKING/modelSale/forklift/'
output_base = r'/WORKING/modelSale/forklift/'

dataset_base = dataset_base.replace('\\', '/')
output_base = output_base.replace('\\', '/')

dataset_images = os.path.join(dataset_base, 'images')
dataset_labels = os.path.join(dataset_base, 'labels')
neg_images = os.path.join(dataset_base, 'negatives')

output_aug_images = os.path.join(output_base, 'aug_images')
output_aug_labels = os.path.join(output_base, 'aug_labels')
output_aug_negs = os.path.join(output_base, 'aug_negatives')

threshold_wh = (9,9)  #min size for augmented box
gen_aug_negatives = False
gen_aug_dataset = True
gen_mosaic_imgs = True
img_aug_count = 1
mosaic_repeat_count = 1

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

def pick_ds_file(ds_files):
    mimg_id = random.randint(0, len(ds_files)-1)
    mimd_file = ds_files[mimg_id]
    mfilename, mfile_extension = os.path.splitext(mimd_file)

    return mimg_id, mfilename, mfile_extension

if __name__ == "__main__":
    augmentation = augment(dataset_images, dataset_labels, neg_images, output_aug_images, output_aug_labels, \
        diverse_1=diverse_1, diverse_2=diverse_2,img_aug_count=img_aug_count, threshold_wh=threshold_wh)

    if gen_aug_negatives == True:
        #augmentation.auto_make(img_aug_count, type_count)
        print("Generate negatives images from ", neg_images)
        #negatives
        for id, file in enumerate(tqdm(os.listdir(neg_images))):
            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()

            if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
                image_path = os.path.join(neg_images, file)

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
                        for way_id in [ 0, 11]:
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
                            elif(way_id == 9):
                                img = augmentation.do_small_larger(img, s_ratio=0.3)
                            elif(way_id == 11):
                                img = augmentation.rgb2gray2rgb(img)


                            ways_txt = str(way_id)

                            #plt.imshow(img) 
                            aug_filename = "{}_{}-{}-{}".format(filename, con_id, ways_txt, count_num)
                            cv2.imwrite(os.path.join(output_aug_negs, aug_filename+file_extension), img)

    if gen_aug_dataset is True:
        print("Generate training images from ", output_aug_negs)

        augmentation.load_dataset(dataset_images)
        augmentation.load_negs(output_aug_negs)
        #Manual
        for id, file in enumerate(tqdm(augmentation.ds_list)):
            #if(id>0):
            #    break

            filename, file_extension = os.path.splitext(file)
            file_extension = file_extension.lower()

            if(file_extension == ".jpg" or file_extension==".jpeg" or file_extension==".png" or file_extension==".bmp"):
                #print("Processing: ", os.path.join(dataset_images, file))

                if os.path.exists(os.path.join(dataset_labels, filename+".xml")):
                    image_path = os.path.join(dataset_images, file)
                    xml_path = os.path.join(dataset_labels, filename+".xml")

                    for count_num in tqdm(range(0, img_aug_count)):
                        
                        #ways = {0: 'no_change', 1:'rotate90', 2:'rotate180', 3:'rotate270', 4:'flip', 5:'shift' }
                        ways = {0: 'no_change', 1:'rotate90', 2:'rotate180', 3:'rotate270' }
                        for con_id in ways:
                            img_org = cv2.imread(image_path)
                            try:
                                test = img_org.shape
                            except:
                                break

                            labelName, bboxes = augmentation.getLabels(image_path, xml_path)

                            cimg, bboxes, labelName = augmentation.get_new_bbox(img_org, bboxes, labelName, ways[con_id])
                            #print('way:', ways[con_id]) 

                            for way_id in tqdm([ 0, 5, 8, 11]):
                            #for way_id in [ 0,1,2,3 ]:
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
                                elif(way_id == 9):
                                    img = augmentation.do_small_larger(img, s_ratio=0.3)
                                elif(way_id == 10):
                                    mid, mfile_name, mfile_ext = pick_ds_file(augmentation.ds_list)
                                    while mfile_name==filename:
                                        mid, mfile_name, mfile_ext = pick_ds_file(augmentation.ds_list)

                                    mimg_path = os.path.join(dataset_images, mfile_name+mfile_ext)
                                    mxml_path = os.path.join(dataset_labels, mfile_name+'.xml')
                                    mlabelName, mbboxes = augmentation.getLabels(mimg_path, mxml_path)
                                    img2 = cv2.imread( mimg_path)
                                    img, (w_ratio, h_ratio) = augmentation.merge_img(img, img2)

                                elif(way_id == 11):
                                    img = augmentation.rgb2gray2rgb(img)

                                ways_txt = str(way_id)

                                #plt.imshow(img) 
                                aug_filename = "{}_{}-{}-{}".format(filename, con_id, ways_txt, count_num)
                                cv2.imwrite(os.path.join(output_aug_images, aug_filename+file_extension), img)

                                aug_labelName, aug_labelXmin, aug_labelYmin, aug_labelXmax, aug_labelYmax = [], [], [], [], []
                                for id, box in enumerate(bboxes):
                                    aug_labelName.append(labelName[id])
                                    aug_labelXmin.append(box[0])
                                    aug_labelYmin.append(box[1])
                                    aug_labelXmax.append(box[2]+box[0])
                                    aug_labelYmax.append(box[3]+box[1])

                                if way_id==10:
                                    for id, box in enumerate(mbboxes):
                                        aug_labelName.append(mlabelName[id])
                                        aug_labelXmin.append( int(box[0] * w_ratio))
                                        aug_labelYmin.append( int(box[1] * h_ratio))
                                        aug_labelXmax.append( int(int(box[2]+box[0]) * w_ratio))
                                        aug_labelYmax.append( int(int(box[3]+box[1]) * h_ratio))


                                xml_file = augmentation.makeDatasetFile(img, file, (aug_labelName, aug_labelXmin, aug_labelYmin, aug_labelXmax, aug_labelYmax))
                                xmlFilename = os.path.join(output_aug_labels, aug_filename + '.xml')
                                file_object = open(xmlFilename, "w")
                                file_object.write(xml_file)
                                file_object.close

    #-------------- 4 splices --------------

    if gen_mosaic_imgs is True:
        for count in range(0, mosaic_repeat_count):
            augmentation.load_augdataset(output_aug_images)
            augmentation.load_augnegs(output_aug_images)

            print('Add 4 images splices') 
            for file in tqdm(augmentation.augds_list):
                augmentation.mosaic_4imgs(file)


