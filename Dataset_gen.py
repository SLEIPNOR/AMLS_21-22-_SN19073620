# -*- coding: utf-8 -*-
"""
Created on Thu Nov 18 23:24:43 2021

@author: Blade
"""

from PIL import Image
import glob
import numpy as np
import matplotlib.pyplot as plt
#from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, IncrementalPCA
import pandas as pd
import gc 
#from sklearn.preprocessing import MinMaxScaler
#%% Import Image
a=glob.glob('./dataset/image/*.jpg') 
Pic_No = a[0]
# print('a=',a,'\ra[0]=',a[0],'\r')
L=len(a)
# img_Mat = np.array([[0]*len(np.array(Image.open(Pic_No).convert('L'),'f'))**2 for i in range(len(a))],float)
img_Mat =[]
#%% Read Image 

for Pic_No in a:
    img = np.array(Image.open(Pic_No).convert('L'),'f')
    # img = img.reshape(img.shape[0]*img.shape[1],1).T
    img = np.array(Image.open(Pic_No).convert('L'),'f').flatten()
    img_Mat.append(img) 
    print('\r'"Reading Process:{0}%".format(round((a.index(Pic_No) + 1) * 100 / len(a))), end="\r",flush=True)
   
img_Mat = np.array(img_Mat)
del img,a,Pic_No,L
gc.collect()
#%% Dataset with label building
label =pd.read_csv('./dataset/label.csv')
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

label = np.array(label['label']).reshape(len(label),1)

# dataset = np.hstack((img_Mat,label)).astype(np.float64)

del i, label_name
gc.collect()
#%% Splitting
x_train, x_test, y_train, y_test = train_test_split(img_Mat, label,random_state=0)

#%% output dataset as csv
# np.savetxt('Dataset.csv', img_Mat, delimiter = ',')
# np.savetxt('label(Type).csv', label, delimiter = ',')
#%% PCA

#Image.fromarray(img_Mat[6].reshape(512,512)).show()   
#plt.imshow(np.hstack((img_Mat[0].reshape(512,512),img_Mat[1].reshape(512,512),img_Mat[2].reshape(512,512))),cmap ='gray') 
# img = np.array(Image.open('./dataset/image/IMAGE_0001.jpg').convert('L'),'f')
# img = img.reshape(img.shape[0]*img.shape[1],1).T


# scaler = MinMaxScaler( )
# scaler.fit(img_Mat[:400,:11700]) 
# scaler.data_max_
# new_img_Mat=scaler.transform(img_Mat[:400,:11700])


# scaler = preprocessing.StandardScaler()
# new_img_Mat = scaler.fit_transform(img_Mat) 

#%% what u should recover

# new_img_Mat = scale(img_Mat, with_mean=True, with_std=True, axis=0)

# pca = PCA(n_components=0.95)
# new_img_Mat_pca = pca.fit_transform(new_img_Mat)

# np.savetxt('PCA_data.csv', new_img_Mat_pca, delimiter = ',')
# np.savetxt('data.csv', img_Mat, delimiter = ',')

#%%

# ipca = IncrementalPCA(n_components=3000)
# new_img_Mat_ipca = ipca.fit_transform(new_img_Mat)
# pca = PCA(n_components= 0.9,svd_solver='full')
# new_img_Mat = pca.transform(img_Mat[:400,:11700])
# print(pca.explained_variance_ratio_)
# X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
# pca1 = PCA(n_components=1)
# newX = pca1.fit_transform(X)

