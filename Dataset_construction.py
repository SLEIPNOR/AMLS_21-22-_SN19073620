# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
from pandas import read_csv

#%%
#Pic_PCA = numpy.loadtxt(open('./PCA_data.csv',"rb"),delimiter=",",skiprows=0)

Pic_PCA = read_csv('./PCA_data.csv',header = None) #pandas read first line of data


#%%
# label = numpy.loadtxt(open('./dataset/label.csv',"rb"),delimiter=",",skiprows=0)
# filename='./dataset/label.csv'
# with open(filename,'rt') as raw_data:
#     data=numpy.loadtxt(raw_data,delimiter=',')
#     print(data.shape)

# label =read_csv('./dataset/label.csv',names=['file_name','label'])

label =read_csv('./dataset/label.csv')
#%%
# label.loc[4,'label']
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
#%%
label = np.array(label['label']).reshape(len(label),1)
#%%
dataset = np.hstack((Pic_PCA,label)) 
np.savetxt('Dataset(type).csv', dataset, delimiter = ',')