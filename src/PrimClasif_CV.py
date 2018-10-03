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

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,f1_score
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

X_train = scalerX.transform(X_df)  
Y_train = Y_df.values

X_testYval_norm = scalerX.transform(X_test_df)  

#Y_norm = scalerY.transform(Y_df)  
#Y_test_norm = scalerY.transform(Y_test_df)  

#Test Val split

X_test_norm,X_val,Y_test,Y_val=train_test_split(X_testYval_norm,
                                            Y_test_df.values,
                                            test_size=0.50,
                                            random_state=45)


X_train=X_train[0:50,:]
Y_train=Y_train[0:50,:]

n_folds=3

""" 
KNN
"""

start_t = time.time()

#n_neighbors_list = list(range(1,30,1))
n_neighbors_list = list(range(1,4,1))


KNN_perf_record_test_CV = {}
KNN_perf_record_train_CV = {}
KNN_perf_record_test = {}
KNN_perf_mean_record_train_CV = {}
KNN_perf_mean_record_train_CV_std = {}
KNN_perf_mean_record_test_CV = {}
KNN_perf_mean_record_test_CV_std = {}
k_fold = StratifiedKFold(n_splits=n_folds)

for n_neighbors in n_neighbors_list:
    fold=-1
    for train_indices, test_indices in k_fold.split(X_train, Y_train):
        fold+=1
        
        print('VAMOS por n_neigbour %d y por la fold %d / %d' % (n_neighbors,fold,n_folds))

        if n_neighbors not in KNN_perf_record_train_CV: KNN_perf_record_train_CV[n_neighbors] = np.zeros(n_folds)
        if n_neighbors not in KNN_perf_record_test_CV: KNN_perf_record_test_CV[n_neighbors] = np.zeros(n_folds)
        if n_neighbors not in KNN_perf_record_test: KNN_perf_record_test[n_neighbors] = np.zeros(n_folds)
        
        X_train_CV, X_test_CV = X_train[train_indices], X_train[test_indices] 
        Y_train_CV, Y_test_CV = Y_train[train_indices], Y_train[test_indices]

    
        knn_clf = neighbors.KNeighborsClassifier(n_neighbors,metric='euclidean')
        knn_clf.fit(X_train_CV, np.ravel(Y_train_CV))
        Y_pred_train_CV=knn_clf.predict(X_train_CV).reshape(-1,1)
        #Y_pred_train_CV=Y_pred_train.reshape(-1,1)
        Y_pred_test_CV=knn_clf.predict(X_test_CV).reshape(-1,1)
        Y_pred_test=knn_clf.predict(X_test_norm).reshape(-1,1)
        
        #perf_record[fold][n_neighbors] = precision_recall_fscore_support(Y_test_CV,Y_pred_test_CV)
        KNN_perf_record_train_CV[n_neighbors][fold] = accuracy_score(Y_train_CV,Y_pred_train_CV)
        KNN_perf_record_test_CV[n_neighbors][fold] = accuracy_score(Y_test_CV,Y_pred_test_CV)
        KNN_perf_record_test[n_neighbors][fold] = accuracy_score(Y_test,Y_pred_test)
    
    if n_neighbors not in KNN_perf_mean_record_train_CV: KNN_perf_mean_record_train_CV[n_neighbors] = np.mean(KNN_perf_record_train_CV[n_neighbors])
    if n_neighbors not in KNN_perf_mean_record_train_CV_std: KNN_perf_mean_record_train_CV_std[n_neighbors] = np.std(KNN_perf_record_train_CV[n_neighbors])
    if n_neighbors not in KNN_perf_mean_record_test_CV: KNN_perf_mean_record_test_CV[n_neighbors] = np.mean(KNN_perf_record_test_CV[n_neighbors])
    if n_neighbors not in KNN_perf_mean_record_test_CV_std: KNN_perf_mean_record_test_CV_std[n_neighbors] = np.std(KNN_perf_record_test_CV[n_neighbors])
    

#calc_error_n_plot(Y_train,Y_pred_train,'TRAIN')
#clasf_report_val['KNN']=calc_error_n_plot(Y_val,Y_pred_val,'VALIDATION')
#clasf_report['KNN']=calc_error_n_plot(Y_test,Y_pred_test,'TEST')

elapsed_t['knn'] = time.time() - start_t


best_index=list(KNN_perf_mean_record_test_CV.keys())[np.argmax(list(KNN_perf_mean_record_test_CV.values()))]

print('KNN - Best\'s train CV accuracy %f (std= %f ) for n_neigbout %d \n' % (np.max(list(KNN_perf_mean_record_train_CV.values())),KNN_perf_mean_record_train_CV_std[best_index],best_index))
print('KNN - Best\'s test CV accuracy %f (std= %f ) for n_neigbout %d \n' % (np.max(list(KNN_perf_mean_record_test_CV.values())),KNN_perf_mean_record_test_CV_std[best_index],best_index))
print('KNN - Test accuracy %s , mean: %f (std= %f) \n' % (KNN_perf_record_test[best_index],np.mean(KNN_perf_record_test[best_index]),np.std(KNN_perf_record_test[best_index])))
print('Time elapsed for kNN %f' % elapsed_t['knn'])

"""
SVM
"""
SVM_perf_record_train_CV = {}
SVM_perf_record_test_CV = {}
SVM_perf_record_test = {}
SVM_perf_mean_record_test_CV = {}
SVM_perf_mean_record_test_CV_std = {}
SVM_perf_mean_record_train_CV = {}
SVM_perf_mean_record_train_CV_std = {}

start_t = time.time()

exp_C=np.arange(-5,11,2)
exp_gammas=np.arange(-15,0,2)

C=np.exp2(exp_C)
gammas=np.exp2(exp_gammas)

config_list=[]
for i in C:
    for j in gammas:
        config_list.append([i,j])

config_idx=-1
for config in config_list:
    config_idx+=1
    fold=-1
    for train_indices, test_indices in k_fold.split(X_train, Y_train):
        fold+=1
        
        print('VAMOS por config %d , %.3f y por la fold %d / %d' % (config[0],config[1],fold,n_folds))

        if config_idx not in SVM_perf_record_train_CV: SVM_perf_record_train_CV[config_idx] = np.zeros(n_folds)
        if config_idx not in SVM_perf_record_test_CV: SVM_perf_record_test_CV[config_idx] = np.zeros(n_folds)
        if config_idx not in SVM_perf_record_test: SVM_perf_record_test[config_idx] = np.zeros(n_folds)

        X_train_CV, X_test_CV = X_train[train_indices], X_train[test_indices] 
        Y_train_CV, Y_test_CV = Y_train[train_indices], Y_train[test_indices]


        clf = SVC(C=config[0],gamma=config[1])

        clf.fit(X_train_CV, np.ravel(Y_train_CV))

        Y_pred_train_CV=clf.predict(X_train_CV).reshape(-1,1)
        Y_pred_test_CV=clf.predict(X_test_CV).reshape(-1,1)
        Y_pred_test=clf.predict(X_test_norm).reshape(-1,1)
        

        SVM_perf_record_train_CV[config_idx][fold] = accuracy_score(Y_train_CV,Y_pred_train_CV)
        SVM_perf_record_test_CV[config_idx][fold] = accuracy_score(Y_test_CV,Y_pred_test_CV)
        SVM_perf_record_test[config_idx][fold] = accuracy_score(Y_test,Y_pred_test)
    
    if config_idx not in SVM_perf_mean_record_train_CV: SVM_perf_mean_record_train_CV[config_idx] = np.mean(SVM_perf_record_train_CV[config_idx])
    if config_idx not in SVM_perf_mean_record_train_CV_std: SVM_perf_mean_record_train_CV_std[config_idx] = np.std(SVM_perf_record_train_CV[config_idx])
    if config_idx not in SVM_perf_mean_record_test_CV: SVM_perf_mean_record_test_CV[config_idx] = np.mean(SVM_perf_record_test_CV[config_idx])
    if config_idx not in SVM_perf_mean_record_test_CV_std: SVM_perf_mean_record_test_CV_std[config_idx] = np.std(SVM_perf_record_test_CV[config_idx])    
    


elapsed_t['SVM'] = time.time() - start_t

#calc_error_n_plot(Y_train,Y_pred_train,'TRAIN')
#calc_error_n_plot(Y_val,Y_pred_val,'VALIDATION')
#clasf_report['SVM']=calc_error_n_plot(Y_test,Y_pred_test,'TEST')


best_index=list(SVM_perf_mean_record_test_CV.keys())[np.argmax(list(SVM_perf_mean_record_test_CV.values()))]

print('SVM - Best\'s train CV accuracy %f (std= %f ) for config %s \n' % (np.max(list(SVM_perf_mean_record_train_CV.values())),SVM_perf_mean_record_train_CV_std[best_index],config_list[best_index]))
print('SVM - Best\'s test CV accuracy %f (std= %f ) for config %s \n' % (np.max(list(SVM_perf_mean_record_test_CV.values())),SVM_perf_mean_record_test_CV_std[best_index],config_list[best_index]))
print('SVM - Best\'s Test accuracy %s , mean: %f (std= %f) \n' % (SVM_perf_record_test[best_index],np.mean(SVM_perf_record_test[best_index]),np.std(SVM_perf_record_test[best_index])))
print('Time elapsed for SVM %f' % elapsed_t['SVM'])



"""
XGBoost
https://xgboost.ai/about
"""

import xgboost as xgb

XGB_perf_record_train_CV = {}
XGB_perf_record_test_CV = {}
XGB_perf_record_test = {}
XGB_perf_mean_record_test_CV = {}
XGB_perf_mean_record_test_CV_std = {}
XGB_perf_mean_record_train_CV = {}
XGB_perf_mean_record_train_CV_std = {}

start_t = time.time()

max_depth_list=np.arange(3,6,1)
eta_list=np.arange(0.05,1,0.05)

config_list=[]
for i in max_depth_list:
    for j in eta_list:
        config_list.append([i,j])

#dtrain = xgb.DMatrix(np.concatenate((X_train,Y_train),axis=1))
#dtest = xgb.DMatrix(np.concatenate((X_test_norm,Y_test_df.values),axis=1))

#dtrain = xgb.DMatrix(X_train, label=Y_train)
#dtest = xgb.DMatrix(X_test_norm,label=Y_test)


config_idx=-1
for config in config_list:
    config_idx+=1
    fold=-1
    for train_indices, test_indices in k_fold.split(X_train, Y_train):
        fold+=1
        
        print('VAMOS por config %d , %.3f y por la fold %d / %d' % (config[0],config[1],fold,n_folds))

        if config_idx not in XGB_perf_record_train_CV: XGB_perf_record_train_CV[config_idx] = np.zeros(n_folds)
        if config_idx not in XGB_perf_record_test_CV: XGB_perf_record_test_CV[config_idx] = np.zeros(n_folds)
        if config_idx not in XGB_perf_record_test: XGB_perf_record_test[config_idx] = np.zeros(n_folds)

        X_train_CV, X_test_CV = X_train[train_indices], X_train[test_indices] 
        Y_train_CV, Y_test_CV = Y_train[train_indices], Y_train[test_indices]

        dtrain_CV = xgb.DMatrix(X_train_CV, label=Y_train_CV)
        dtest_CV = xgb.DMatrix(X_test_CV,label=Y_test_CV)
        dtest = xgb.DMatrix(X_test_norm,label=Y_test)

        # specify parameters via map
        param = {'max_depth':config[0], 'eta':config[1], 'silent':1, 'objective':'multi:softmax', 'num_class':5 }
        num_round = 2
        bst = xgb.train(param, dtrain_CV, num_round)
        # make prediction
        #Y_pred_test = bst.predict(dtest)
        
        Y_pred_train_CV=bst.predict(dtrain_CV)
        Y_pred_test_CV=bst.predict(dtest_CV)
        Y_pred_test=bst.predict(dtest)
        

        XGB_perf_record_train_CV[config_idx][fold] = accuracy_score(Y_train_CV,Y_pred_train_CV)
        XGB_perf_record_test_CV[config_idx][fold] = accuracy_score(Y_test_CV,Y_pred_test_CV)
        XGB_perf_record_test[config_idx][fold] = accuracy_score(Y_test,Y_pred_test)
    
    if config_idx not in XGB_perf_mean_record_train_CV: XGB_perf_mean_record_train_CV[config_idx] = np.mean(XGB_perf_record_train_CV[config_idx])
    if config_idx not in XGB_perf_mean_record_train_CV_std: XGB_perf_mean_record_train_CV_std[config_idx] = np.std(XGB_perf_record_train_CV[config_idx])
    if config_idx not in XGB_perf_mean_record_test_CV: XGB_perf_mean_record_test_CV[config_idx] = np.mean(XGB_perf_record_test_CV[config_idx])
    if config_idx not in XGB_perf_mean_record_test_CV_std: XGB_perf_mean_record_test_CV_std[config_idx] = np.std(XGB_perf_record_test_CV[config_idx])    
    


elapsed_t['XGB'] = time.time() - start_t



best_index=list(XGB_perf_mean_record_test_CV.keys())[np.argmax(list(XGB_perf_mean_record_test_CV.values()))]

print('XGB - Best\'s train CV accuracy %f (std= %f ) for config %s \n' % (np.max(list(XGB_perf_mean_record_train_CV.values())),XGB_perf_mean_record_train_CV_std[best_index],config_list[best_index]))
print('XGB - Best\'s test CV accuracy %f (std= %f ) for config %s \n' % (np.max(list(XGB_perf_mean_record_test_CV.values())),XGB_perf_mean_record_test_CV_std[best_index],config_list[best_index]))
print('XGB - Best\'s Test accuracy %s , mean: %f (std= %f) \n' % (XGB_perf_record_test[best_index],np.mean(XGB_perf_record_test[best_index]),np.std(XGB_perf_record_test[best_index])))
print('Time elapsed for XGB %f' % elapsed_t['XGB'])



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


clasf_report['DNN']=calc_error_n_plot(Y_test,np.argmax(Y_pred_test,axis=1),'TEST')

"""
Generate report
"""
final_report=''
final_report=' \t             precision    recall  f1-score   support \n'
for model in clasf_report.keys():
    report=clasf_report[model].split('\n')
    final_report+=model+'\t'+report[5+3]+' \n'
print(final_report)
