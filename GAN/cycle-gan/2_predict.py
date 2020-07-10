import cv2
import numpy as np
import glob
from keras.models import load_model
from numpy import load
from numpy import vstack
from matplotlib import pyplot
from numpy.random import randint
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

np_dataset = '/DATA1/Datasets_download/GAN/horse2zebra/horse2zebra_256.npz'
A_path = '/DATA1/Datasets_download/GAN/horse2zebra/demo/A/*.jpg'
B_path = '/DATA1/Datasets_download/GAN/horse2zebra/demo/B/*.jpg'
img_resize = (256,256)

def load_real_samples(filename):

    X1 = []
    for file in glob.glob(A_path):
        print(file)
        img = cv2.imread(file)
        print(img.shape)
        img = cv2.resize(img, (256,256))
        print(img.shape)
        X1.append(img)

    X2 = []
    for file in glob.glob(B_path):
        img = cv2.imread(file)
        img = cv2.resize(img, (256,256))
        X2.append(img)

    X1 = np.array(X1)
    X2 = np.array(X2)
    # scale from [0,255] to [-1,1]
    X1 = (X1 - 127.5) / 127.5
    X2 = (X2 - 127.5) / 127.5
    return [X1, X2]

# select a random sample of images from the dataset
def select_sample(dataset, n_samples):
    # choose random instances
    ix = randint(0, dataset.shape[0], n_samples)
    # retrieve selected images
    X = dataset[ix]
    return X

# load dataset
A_data, B_data = load_real_samples(np_dataset)
print('Loaded', A_data.shape, B_data.shape)

# load the models
cust = {'InstanceNormalization': InstanceNormalization}
model_AtoB = load_model('g_model_AtoB_106830.h5', cust)
model_BtoA = load_model('g_model_BtoA_106830.h5', cust)

# plot the image, the translation, and the reconstruction
def show_plot(imagesX, imagesY1, imagesY2):
    images = vstack((imagesX, imagesY1, imagesY2))
    titles = ['Real', 'Generated', 'Reconstructed']
    # scale from [-1,1] to [0,1]
    images = (images + 1) / 2.0

    '''
        # plot images row by row
        for i in range(len(images)):
           # define subplot
            pyplot.subplot(1, len(images), 1 + i)
            # turn off axis
            pyplot.axis('off')
            # plot raw pixel data
            pyplot.imshow(images[i])
            # title
            pyplot.title(titles[i])

            pyplot.show()
    '''

# plot A->B->A
#A_real = select_sample(A_data, 10)
A_real = A_data
B_generated  = model_AtoB.predict(A_real)
A_reconstructed = model_BtoA.predict(B_generated)

for i in range(0, len(A_real)):
    A0, AB0, BA0  = A_real[i], B_generated[i], A_reconstructed[i]
    A0 = 256 * ((A0 + 1) / 2.0)
    AB0 = 256 * ((AB0 + 1) / 2.0)
    BA0 = 256 * ((BA0 + 1) / 2.0)

    cv2.imwrite(str(i) + "_A0.jpg", A0)
    cv2.imwrite(str(i) + "_AB0.jpg", AB0)
    cv2.imwrite(str(i) + "_BA0.jpg", BA0)


#show_plot(A_real, B_generated, A_reconstructed)
# plot B->A->B
#B_real = select_sample(B_data, 10)
B_real = B_data
A_generated  = model_BtoA.predict(B_real)
B_reconstructed = model_AtoB.predict(A_generated)

#show_plot(B_real, A_generated, B_reconstructed)
print("B_real:", B_real.shape)
print("A_generated:", A_generated.shape)
print("B_reconstructed:", B_reconstructed.shape)


#B1, BA1, AB1  = B_real[0], A_generated[0], B_reconstructed[0]

for i in range(0, len(B_real)):
    B1, BA1, AB1  = B_real[i], A_generated[i], B_reconstructed[i]
    B1 = 256 * ((B1 + 1) / 2.0)
    BA1 = 256 * ((BA1 + 1) / 2.0)
    AB1 = 256 * ((AB1  + 1) / 2.0)

    cv2.imwrite(str(i) + "_B1.jpg", B1)
    cv2.imwrite(str(i) + "_BA1.jpg", BA1)
    cv2.imwrite(str(i) + "_AB1.jpg", AB1)

