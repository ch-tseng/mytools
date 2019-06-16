from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
import numpy as np

num_aug = 1
gen_size = (300,300)
#ds_folder = "/home/digits/datasets/Diabnext_org2_dataset"
ds_folder = "/home/digits/datasets/Diabnext_277_classes_org2"
TRAIN_NUMBER_PER_CLASS = 800

#---------------------------------------------------------------------

datagen = ImageDataGenerator(
        featurewise_center=False, featurewise_std_normalization=False,
        samplewise_center=False, samplewise_std_normalization=False,
        #zca_whitening=True,
        rotation_range=60,
        width_shift_range=0.2, height_shift_range=0.2,
        shear_range=0.35,
        fill_mode='constant', zoom_range=0.5,
        channel_shift_range=10,
        horizontal_flip=True,
        vertical_flip=True
        #rescale= 1/255
    )

'''
gen_data = datagen.flow_from_directory(ds_folder,
                                       class_mode=None,
                                       batch_size=2,
                                       shuffle=False,
                                       save_to_dir=ds_folder,
                                       save_prefix='aug',
                                       target_size=gen_size)

'''

num_img = 0

for classname in os.listdir(ds_folder):
    full_path = os.path.join(ds_folder, classname)

    #folder or file?
    if(os.path.isdir(full_path)):
        imgfiles = os.listdir(full_path)
        num_class_imgs = len(imgfiles)
        var_imgs = TRAIN_NUMBER_PER_CLASS - num_class_imgs
        print("=== CLASS:{}, images:{}, need more:{} ===".format(classname, num_class_imgs, var_imgs))

        ii = 0
        num_aug_imgs = 0
        while num_aug_imgs<var_imgs:
            for imgfile in imgfiles:

                filename, file_extension = os.path.splitext(imgfile)
                file_extension_lower = file_extension.lower()

                try:
                    img = cv2.imread(os.path.join(full_path, imgfile))
                    img = img[np.newaxis, :]

                except:
                    print("warnning: image read error:", os.path.join(full_path, imgfile))
                    continue

                i = 0
                print(num_img, os.path.join(full_path, imgfile))
                num_img += 1
                #for batch in datagen.flow(img, batch_size=num_aug, save_to_dir=SAVE_PATH, save_prefix='zaug', save_format='png'):
                for batch in datagen.flow(img, batch_size=num_aug):
                    aug_file = os.path.join(full_path, filename+"-aug"+str(ii).zfill(2)+'-'+str(i).zfill(2)+file_extension)
                    print("   ---> save to", aug_file)
                    cv2.imwrite(aug_file, batch[0].astype(np.uint8))
                    i += 1
                    num_aug_imgs += 1

                    if (i>=num_aug) or (num_aug_imgs>=var_imgs):
                        break

                if(num_aug_imgs>=var_imgs):
                    break

            ii += 1
