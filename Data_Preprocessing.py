# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 00:52:34 2021

@author: Blade
"""

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
import gc 

def dataset_Gen(root_feature,root_label, dim, method):
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
            label.loc[i,'label'] = 2
            
        elif label_name == 'pituitary_tumor':
            label.loc[i,'label'] = 3
        i = i+1
        
    label = np.array(label['label'],float).ravel()
        
    # Splitting
    x_train, x_test, y_train, y_test = train_test_split(img_Mat, label, test_size=0.25, random_state=0)
    del img_Mat, label
    gc.collect()
    
    # Standardize  
    print('Standardization...')
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train) 
    x_test = scaler.transform(x_test) 
    
    # x_train = scale(x_train, with_mean=True, with_std=True, axis=0)
    # x_test = scale(x_train, with_mean=True, with_std=True, axis=0)
    
    # scaler = MinMaxScaler()
    # x_train = scaler.fit_transform(x_train)
    # x_test = scaler.transform(x_test)
    
    
    # LDA
    if method == 'LDA':
        print('LDA Implemented...')
        lda = LinearDiscriminantAnalysis()
        x_train = lda.fit_transform(x_train,y_train)
        # x_train = lda.transform(x_train)
        x_test = lda.transform(x_test)
        
        print('Writing Data...')
        np.savetxt('x_train_lda.csv', x_train, delimiter = ',')
        np.savetxt('x_test_lda.csv', x_test, delimiter = ',')
        np.savetxt('y_train.csv', y_train, delimiter = ',')
        np.savetxt('y_test.csv', y_test, delimiter = ',')
    
    # PCA
    elif method == 'PCA':
        print('PCA Implemented...')
        pca = PCA(n_components = dim)
        x_train = pca.fit_transform(x_train)
        # x_train = pca.transform(x_train)
        x_test = pca.transform(x_test)
        
        Variance = pca.explained_variance_ratio_
        SValue = pca.singular_values_
        
        # Vcomp = pca.components_
        # Cov_train = np.dot(x_train.T,x_train)
        # Cov_test = np.dot(x_test.T,x_test)
        # x_test = np.dot(Vcomp,x_test.T).T
        
        
        # Writing
        print('Writing Data...')
        np.savetxt('x_train_pca.csv', x_train, delimiter = ',')
        np.savetxt('x_test_pca.csv', x_test, delimiter = ',')
        np.savetxt('y_train.csv', y_train, delimiter = ',')
        np.savetxt('y_test.csv', y_test, delimiter = ',')
        # np.savetxt('Vcomp.csv', Vcomp, delimiter = ',')
        np.savetxt('Variance.csv', Variance, delimiter = ',')
        np.savetxt('SValue.csv', SValue, delimiter = ',')
        
    print('Data Preprocessing Completion!')
    return 


method = 'LDA'
root_feature = './dataset/image/*.jpg'
root_label = './dataset/label.csv'
dim = 0.95
dataset_Gen(root_feature,root_label, dim, method)
# Variance, SValue, Vcomp, x_train_pca, x_test, y_train, y_test  = dataset_Gen(root_feature,root_label, dim)
