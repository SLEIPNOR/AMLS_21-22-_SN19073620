# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 00:52:34 2021

@author: Z.Han
"""
"""This code is for image preprocessing including 5-fold cross-validation set splitting 
and PCA, LDA (we not use here as it cannot give good results). Also, for formal training, 
this code also preprocessed 3000 training images and 200 test images.

NOTE HERE AS THE PREPROCESSING SPEND TOO MUCH TIME,ALL PREPROCESSED DATA WERE ALREADY 
GENERATED FOR YOU TO USE, SO YOU CAN RUN THE MODEL DIRECTLY 

if you want to preprocess data by yourself, you must put the dataset (located at Task_B fold)
fold in the directory."""

from PIL import Image
import glob
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pandas as pd
from sklearn.model_selection import KFold
import gc 

# load data for 5-fold cross-validation
def dataset_Gen(root_feature,root_label, dim):
    # Grabbing all jpg img filename in folder
    print('Grabbing All jpg...')
    Pic_jpg = glob.glob(root_feature) 
    Pic_No = Pic_jpg[0]
    img_Mat = []
    # Pic reading 
    for Pic_No in Pic_jpg:
        img = np.array(Image.open(Pic_No).convert('L'),'f')
        img = np.array(Image.open(Pic_No).convert('L'),'f').flatten()
        img_Mat.append(img) 
        print('\r'"Pic Reading Process:{0}%".format(round((Pic_jpg.index(Pic_No) + 1) * 100 / len(Pic_jpg))), end="",flush=True)
    img_Mat = np.array(img_Mat)
   
    # Input label
    print('\nLabel Generation...')
    label =pd.read_csv(root_label)
    
    # Transform label into num
    i = 0
    for label_name in label['label']:
        if label_name == 'no_tumor':
            label.loc[i,'label'] = 0
            
        elif label_name == 'meningioma_tumor':
            label.loc[i,'label'] = 1
            
        elif label_name == 'glioma_tumor':
            label.loc[i,'label'] = 1
            
        elif label_name == 'pituitary_tumor':
            label.loc[i,'label'] = 1
        i = i+1
        
    label = np.array(label['label'],float).ravel()
        
    # Splitting
    
    kf=KFold(n_splits=5,random_state=0,shuffle=True)
    i = 0
    for train_index,valid_index in kf.split(img_Mat):
        x_train, x_val, y_train, y_val = img_Mat[train_index], img_Mat[valid_index],label[train_index], label[valid_index]
    # x_train, x_test, y_train, y_test = train_test_split(img_Mat, label, test_size=0.25, random_state=0)
    # del img_Mat, label
    # gc.collect()
    
        # Standardize  
        print('Standardization...')
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train) 
        x_val = scaler.transform(x_val) 
        
        # x_train = scale(x_train, with_mean=True, with_std=True, axis=0)
        # x_test = scale(x_train, with_mean=True, with_std=True, axis=0)
        
        # scaler = MinMaxScaler()
        # x_train = scaler.fit_transform(x_train)
        # x_test = scaler.transform(x_test)
    
    
        # LDA
      
        # print('LDA Implemented...')
        # lda = LinearDiscriminantAnalysis()
        # x_train = lda.fit_transform(x_train,y_train)
        # # x_train = lda.transform(x_train)
        # x_test = lda.transform(x_test)
        
        # print('Writing Data...')
        # np.savetxt('x_train_lda.csv', x_train, delimiter = ',')
        # np.savetxt('x_test_lda.csv', x_test, delimiter = ',')
        # np.savetxt('y_train.csv', y_train, delimiter = ',')
        # np.savetxt('y_test.csv', y_test, delimiter = ',')
        
        # PCA
        
        print('PCA Implemented...')
        pca = PCA(n_components = dim)
        x_train = pca.fit_transform(x_train)
        # x_train = pca.transform(x_train)
        x_val = pca.transform(x_val)
        
        # Variance = pca.explained_variance_ratio_
        # SValue = pca.singular_values_
        
        # Vcomp = pca.components_
        # Cov_train = np.dot(x_train.T,x_train)
        # Cov_test = np.dot(x_test.T,x_test)
        # x_test = np.dot(Vcomp,x_test.T).T
        
        
        # Writing
        print('Writing Data...')
        np.savetxt('x_train_pca_{}.csv'.format(i), x_train, delimiter = ',')
        np.savetxt('x_test_pca_{}.csv'.format(i), x_val, delimiter = ',')
        np.savetxt('y_train_{}.csv'.format(i), y_train, delimiter = ',')
        np.savetxt('y_test_{}.csv'.format(i), y_val, delimiter = ',')
        # np.savetxt('Vcomp.csv', Vcomp, delimiter = ',')
        # np.savetxt('Variance.csv', Variance, delimiter = ',')
        # np.savetxt('SValue.csv', SValue, delimiter = ',')
        print('\r'"Fold {} Data Preprocessing Completion!".format(i+1), end="",flush=True)   
        
        i +=1
    return 

# load data for formal training

def data_load(root_feature, root_label):
    Pic_jpg = glob.glob(root_feature)
    Pic_No = Pic_jpg[0]
    img_Mat = []
    # Pic reading
    for Pic_No in Pic_jpg:
        img = np.array(Image.open(Pic_No).convert('L'), 'f')
        img = np.array(Image.open(Pic_No).convert('L'), 'f').flatten()
        img_Mat.append(img)
        print('\r'"Pic Reading Process:{0}%".format(round((Pic_jpg.index(Pic_No) + 1) * 100 / len(Pic_jpg))), end="",
              flush=True)
    img_Mat = np.array(img_Mat)

    # Input label
    print('\nLabel Generation...')
    label = pd.read_csv(root_label)

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

    label = np.array(label['label'], float).ravel()

    return img_Mat, label



#%% five-fold cross-validation dataset

root_feature = './dataset/image/*.jpg'
root_label = './dataset/label.csv'
dim = 0.95
dataset_Gen(root_feature,root_label, dim)

# Variance, SValue, Vcomp, x_train_pca, x_test, y_train, y_test  = dataset_Gen(root_feature,root_label, dim)

#%% formal training dataset & test dataset

# loading training data
X_train, y_train = data_load('./dataset/image/*.jpg', './dataset/label.csv')

# loading test data
X_test, y_test = data_load('./dataset/test/image/*.jpg', './dataset/test/label.csv')

print('Standardization...')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print('PCA Implemented...')
pca = PCA(n_components = 0.95)
X_train = pca.fit_transform(X_train)
# x_train = pca.transform(x_train)
X_test = pca.transform(X_test)

np.savetxt('x_train_pca_formal.csv', X_train , delimiter = ',')
np.savetxt('x_test_pca_formal.csv', X_test, delimiter = ',')
np.savetxt('y_train_formal.csv', y_train, delimiter = ',')
np.savetxt('y_test_formal.csv', y_test, delimiter = ',')