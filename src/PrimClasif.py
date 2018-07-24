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

prefijo=''
path_datos=('../data'+prefijo+'/')
path_results=('../results'+prefijo+'/')

X_df=pd.read_csv(path_datos+'XTrn.txt',header=None,usecols=range(5))
Y_df=pd.read_csv(path_datos+'YTrn.txt',header=None,usecols=range(1))
Xtest_df=pd.read_csv(path_datos+'XTest.txt',header=None,usecols=range(5))
Ytest_df=pd.read_csv(path_datos+'YTest.txt',header=None,usecols=range(1))
