# -*- coding: utf-8 -*-
"""
Created on Sat Nov 20 22:28:30 2021

@author: Blade
"""
import dask.dataframe as dd
import dask.array as da
from dask_ml.model_selection import train_test_split
from dask_ml.decomposition import PCA
from dask_ml.preprocessing import MinMaxScaler
import numpy as np
import gc 
#%% Import data
dataset = dd.read_csv('./Dataset(Type).csv',header = None)
#%% Splitting

X = dataset.drop(dataset.shape[1]-1,axis=1)
Y = dataset[dataset.shape[1]-1].to_frame()

x_train, x_test, y_train, y_test = train_test_split(X, Y,random_state=0, shuffle=True)

# x_train = x_train.compute()
# x_test = x_test.compute()
# y_train = y_train.compute()
# y_test = y_test.compute()

del dataset, X, Y
gc.collect()
#%% PCA Implemented
x_train = x_train.to_dask_array(lengths=True)

scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

pca = PCA(n_components = 400,svd_solver='auto')
x_train = pca.fit_transform(x_train)

x_train = x_train.compute()
Variance = pca.explained_variance_ratio_
SValue = pca.singular_values_
Vcomp = pca.components_