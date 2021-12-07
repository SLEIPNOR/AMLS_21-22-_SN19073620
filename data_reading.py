from PIL import Image
import glob
import numpy as np
import pandas as pd

def tumor_4class():
    Pic_jpg = glob.glob('./dataset/image/*.jpg')
    Pic_No = Pic_jpg[0]
    img_Mat = []
    # Pic reading
    for Pic_No in Pic_jpg:
        img = np.array(Image.open(Pic_No).convert('L'), 'f')
        img = np.array(Image.open(Pic_No).convert('L'), 'f')
        img_Mat.append(img)
        print('\r'"loading images:{0}%".format(round((Pic_jpg.index(Pic_No) + 1) * 100 / len(Pic_jpg))), end="",
              flush=True)
    img_Mat = np.array(img_Mat)
    img_Mat = img_Mat.reshape(img_Mat.shape[0], img_Mat.shape[1], img_Mat.shape[2], 1)


    # Input label
    label = pd.read_csv('dataset/label.csv')

    # Transform label into num
    i = 0
    for label_name in label['label']:
        if label_name == 'no_tumor':
            label.loc[i, 'label'] = 0

        elif label_name == 'meningioma_tumor':
            label.loc[i, 'label'] = 1

        elif label_name == 'glioma_tumor':
            label.loc[i, 'label'] = 2

        elif label_name == 'pituitary_tumor':
            label.loc[i, 'label'] = 3
        i = i + 1

    label = np.array(label['label'], int)
    label = label.reshape(label.shape[0], 1)
    return img_Mat, label


def tumor_2class():
    Pic_jpg = glob.glob('./dataset/image/*.jpg')
    Pic_No = Pic_jpg[0]
    img_Mat = []
    # Pic reading
    for Pic_No in Pic_jpg:
        img = np.array(Image.open(Pic_No).convert('L'), 'f')
        img = np.array(Image.open(Pic_No).convert('L'), 'f')
        img_Mat.append(img)
        print('\r'"loading images:{0}%".format(round((Pic_jpg.index(Pic_No) + 1) * 100 / len(Pic_jpg))), end="",
              flush=True)
    img_Mat = np.array(img_Mat)
    img_Mat = img_Mat.reshape(img_Mat.shape[0], img_Mat.shape[1], img_Mat.shape[2], 1)
    # Input label
    label = pd.read_csv('dataset/label.csv')

    # Transform label into num
    i = 0
    for label_name in label['label']:
        if label_name == 'no_tumor':
            label.loc[i, 'label'] = 0

        elif label_name == 'meningioma_tumor':
            label.loc[i, 'label'] = 1

        elif label_name == 'glioma_tumor':
            label.loc[i, 'label'] = 1

        elif label_name == 'pituitary_tumor':
            label.loc[i, 'label'] = 1
        i = i + 1

    label = np.array(label['label'], int)
    label = label.reshape(label.shape[0], 1)
    return img_Mat, label



