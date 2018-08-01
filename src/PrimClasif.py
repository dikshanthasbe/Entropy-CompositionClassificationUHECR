#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 09:55:10 2018

@author: alberto
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD,Adam
from keras.metrics import  categorical_accuracy
from keras.wrappers.scikit_learn import KerasRegressor
from keras import initializers
from keras import callbacks
from keras.utils.np_utils import to_categorical

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score


from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

import time

prefijo=''
path_datos=('../data'+prefijo+'/')
path_results=('../results'+prefijo+'/')

random_st=42 
seed = 7 
np.random.seed(seed)

#X_df=pd.read_csv(path_datos+'XTrn.txt',usecols=['NALLParticlesTotal','MUTotal','ELTotal','Zenith','Energy'])
X_df=pd.read_csv(path_datos+'XTrn.txt',sep='  ',header=None)
Y_df=pd.read_csv(path_datos+'YTrn.txt',sep='  ',header=None)
Xtest_df=pd.read_csv(path_datos+'XTest.txt',sep='  ',header=None)
Ytest_df=pd.read_csv(path_datos+'YTest.txt',sep='  ',header=None)

#TODO Train Val split


""" 
KNN
"""

start_t_knn = time.time()


elapsed_t_knn = time.time() - start_t_knn

print('Time elapsed for kNN %f', elapsed_t_knn)