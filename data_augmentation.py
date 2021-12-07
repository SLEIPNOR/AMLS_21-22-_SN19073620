
""" self made module by @ Z.Han"""

"""data_augmentation.py module provides down_sampling & 
data_augmentation for data preprocessing prior to training, 
data_trans func is used in data_augmentation as a called function
"""


import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import data_reading as dr
import numpy as np
from random import uniform as un
from PIL import Image

def down_sampling(images, size):

    images_d = []
    for i in range(len(images)):
        images_d.append(np.array(tf.image.resize(images[i], [size, size])))
        print('\r'"down sampling:{0}%".format(round((i + 1) * 100 / len(images)))
              , end="", flush=True)
    images_d = np.array(images_d)
    return images_d




def data_trans(img,crop_size,resize_size):
    size = img.shape
    img = tf.image.random_crop(img[:,:,0],[crop_size[0],crop_size[1]])
    img = tf.image.resize(np.reshape(img, [crop_size[0], crop_size[1], 1])
                          , [resize_size[0], resize_size[1]])
    img = tf.image.resize_with_pad(np.reshape(img, [resize_size[0], resize_size[1], 1])
                                   , size[0], size[1])
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, 0.5)
    img = tf.image.random_contrast(img, 0.2, 1.8)

    return img

def data_aug (images,labels,crop_min_max,export):
    labels_aug = []
    images_aug = []
    for i in range(len(images)):
        image_new = data_trans(images[i]
                               ,[int(un(crop_min_max[0],crop_min_max[1]))
                                   ,int(un(crop_min_max[0], crop_min_max[1]))]
                               ,[int(un(crop_min_max[0],crop_min_max[1]))
                                   ,int(un(crop_min_max[0], crop_min_max[1]))])
        images_aug.append(image_new)
        labels_aug.append(labels[i][0])
        if export:

            im = Image.fromarray(np.reshape(np.array(image_new), [512,512])).convert("L")
            im.save('./dataset/augmentation_data/AugImg_{}.jpg'.format(i))

        print('\r'"data augmentation:{0}%".format(round((i + 1) * 100 / len(images)))
              , end="", flush=True)
    images_aug = np.array(images_aug)
    labels_aug = np.array(labels_aug, ndmin=2).T
    if export:
        np.savetxt('./dataset/augmentation_data/aug_label.csv', labels_aug, delimiter = ',')

    return images_aug, labels_aug








#%%
# images, labels = dr.tumor_4class()
# images = down_sampling(images,128)

#%%
# images_aug, labels_aug = data_aug (images,labels, crop_min_max = [400,500],export = False)

#%%

# print((label==labels).all())
# images = np.concatenate((images, np.reshape(image_new, [1, image_new.shape[0], image_new.shape[1], 1])), axis=0)
# images = np.concatenate((images, images), axis=0)
# label = np.concatenate((label, np.reshape(labels[i], [1, 1])), axis=0)
#%% plot

#
# i = 666
# class_names = ['no_tumor', 'meningioma_tumor', 'glioma_tumor', 'pituitary_tumor']
#
# plt.figure(figsize=(10, 10))
#
# # plt.subplot(2, 2, 1)
# # plt.imshow(images_aug[i], cmap ='gray')
# # plt.xlabel(class_names[labels_aug[i][0]])
#
# plt.subplot(2, 2, 2)
# plt.imshow(images[i], cmap ='gray')
# plt.xlabel(class_names[labels[i][0]])
#
# plt.show()


