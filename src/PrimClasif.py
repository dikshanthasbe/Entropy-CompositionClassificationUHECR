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

from keras.utils import plot_model
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

# Classifiers used
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from plot_conf_matrix import plot_conf_matrix

from tools import calc_error_n_plot

import time

elapsed_t={}
clasf_report={}
prefijo=''
path_datos=('../data'+prefijo+'/')
path_results=('../results'+prefijo+'/')

random_st=42 
seed = 7 
np.random.seed(seed)

#X_df=pd.read_csv(path_datos+'XTrn.txt',usecols=['NALLParticlesTiotal','MUTotal','ELTotal','Zenith','Energy'])
X_df=pd.read_csv(path_datos+'XTrn.txt',sep='  ',header=None)
Y_df=pd.read_csv(path_datos+'YTrn.txt',sep='  ',header=None)
X_test_df=pd.read_csv(path_datos+'XTest.txt',sep='  ',header=None)
Y_test_df=pd.read_csv(path_datos+'YTest.txt',sep='  ',header=None)


scalerX = StandardScaler()  
scalerX.fit(X_df)  

#scalerY = StandardScaler()  
#scalerY.fit(Y_df) 

X_norm = scalerX.transform(X_df)  
X_test_norm = scalerX.transform(X_test_df)  

#Y_norm = scalerY.transform(Y_df)  
#Y_test_norm = scalerY.transform(Y_test_df)  

#Train Val split

X_train,X_val,Y_train,Y_val=train_test_split(X_norm,
                                            Y_df.values,
                                            test_size=0.20,
                                            random_state=45)

"""
BOOOORRAME!!!!!!
"""
X_train=X_train[0:500,:]
Y_train=Y_train[0:500,:]

"""
-----------------------------
"""


""" 
KNN
"""

start_t = time.time()

n_neighbors = 3
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(X_train, np.ravel(Y_train))

elapsed_t['knn'] = time.time() - start_t

Y_pred_train=clf.predict(X_train)
Y_pred_train=Y_pred_train.reshape(-1,1)
Y_pred_val=clf.predict(X_val)
Y_pred_val=Y_pred_val.reshape(-1,1)
Y_pred_test=clf.predict(X_test_norm)
Y_pred_test=Y_pred_test.reshape(-1,1)

calc_error_n_plot(Y_train,Y_pred_train,'TRAIN')
calc_error_n_plot(Y_val,Y_pred_val,'VALIDATION')
clasf_report['KNN']=calc_error_n_plot(Y_test_df.values,Y_pred_test,'TEST')


print('Time elapsed for kNN %f' % elapsed_t['knn'])


"""
SVM
"""

start_t = time.time()

C=0.75
clf = SVC(C)

clf.fit(X_train, np.ravel(Y_train))

elapsed_t['SVM'] = time.time() - start_t

Y_pred_train=clf.predict(X_train)
Y_pred_train=Y_pred_train.reshape(-1,1)
Y_pred_val=clf.predict(X_val)
Y_pred_val=Y_pred_val.reshape(-1,1)
Y_pred_test=clf.predict(X_test_norm)
Y_pred_test=Y_pred_test.reshape(-1,1)

calc_error_n_plot(Y_train,Y_pred_train,'TRAIN')
calc_error_n_plot(Y_val,Y_pred_val,'VALIDATION')
clasf_report['SVM']=calc_error_n_plot(Y_test_df.values,Y_pred_test,'TEST')

"""
Random Forest
"""
clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
clf.fit(X_train, np.ravel(Y_train))

elapsed_t['RandForest'] = time.time() - start_t

Y_pred_train=clf.predict(X_train)
Y_pred_train=Y_pred_train.reshape(-1,1)
Y_pred_val=clf.predict(X_val)
Y_pred_val=Y_pred_val.reshape(-1,1)
Y_pred_test=clf.predict(X_test_norm)
Y_pred_test=Y_pred_test.reshape(-1,1)

calc_error_n_plot(Y_train,Y_pred_train,'TRAIN')
calc_error_n_plot(Y_val,Y_pred_val,'VALIDATION')
clasf_report['RandForest']=calc_error_n_plot(Y_test_df.values,Y_pred_test,'TEST')

"""
XGBoost
https://xgboost.ai/about
"""

import xgboost as xgb

#dtrain = xgb.DMatrix(np.concatenate((X_train,Y_train),axis=1))
#dtest = xgb.DMatrix(np.concatenate((X_test_norm,Y_test_df.values),axis=1))

dtrain = xgb.DMatrix(X_train, label=Y_train)
dtest = xgb.DMatrix(X_test_norm,label=Y_test_df.values)

# specify parameters via map
param = {'max_depth':10, 'eta':1, 'silent':1, 'objective':'multi:softmax', 'num_class':5 }
num_round = 2
bst = xgb.train(param, dtrain, num_round)
# make prediction
Y_pred_test = bst.predict(dtest)

clasf_report['XGB']=calc_error_n_plot(Y_test_df.values,Y_pred_test,'TEST')

"""
DNN
"""
import keras.utils

def clasif_model():
    individual=[(150, 0), (100, 0), (100,0), (50,0)]

    activation_functions={
        0:"relu",
        1:"sigmoid",
        2:"softmax",
        3:"tanh",
        #4:"selu", 
        4:"softplus",
        #6:"softsign",
        5:"linear"
    }

    dimension=5
    model = Sequential()
    for units,activ_f in individual:       
       if(units>5): 
           model.add(Dense(units=units, input_dim=dimension, kernel_initializer=initializers.Constant(value=0.025), activation=activation_functions[activ_f]))
    
    model.add(Dense(units=5, activation="softmax", kernel_initializer=initializers.Constant(value=0.025)))   

    #SGD(lr=0.05, momentum=0.1, decay=0.001, nesterov=False)
    #model.compile(loss='mean_squared_error', optimizer='sgd')
      
    #Adam(lr=0.1, beta_1=0.09, beta_2=0.999, epsilon=1e-08, decay=0.0) #adam defaults Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    #Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0) # Adan defaults
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model


#Prepare one hot encoding...
hot_Y_train = keras.utils.to_categorical(Y_train,num_classes=5)
hot_Y_val = keras.utils.to_categorical(Y_val,num_classes=5)

estimator = KerasRegressor(build_fn=clasif_model, nb_epoch=100, verbose=1)
estimator.fit(X_train,hot_Y_train, epochs=150, batch_size=20, validation_data=(X_val,hot_Y_val))
Y_pred_test = estimator.predict(X_test_norm)


clasf_report['DNN']=calc_error_n_plot(Y_test_df.values,np.argmax(Y_pred_test,axis=1),'TEST')
