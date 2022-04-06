# -*- coding: utf-8 -*-
## adaboost
#%% library
import numpy as np
import pandas as pd
import sys
sys.path.append(r'G:\project\Machine Learning in Action\My_Machine Learning in Action\chapter7 AdaBoost')

from adaboost_functions import *
#%% test adaboost 
x,y=load_Simple_Data()

adaboostClassifier=adaBoostTrain(x,y,40)
adaClassify(np.mat([[1,2],[1,4],[-1,1]]),adaboostClassifier)

#%% load datasets
training_set= pd.read_table("G:\project\Machine Learning in Action\My_Machine Learning in Action\chapter7 AdaBoost\horseColicTraining2.txt",
                 sep='\t',header=None,index_col=None)
testing_set= pd.read_table("G:\project\Machine Learning in Action\My_Machine Learning in Action\chapter7 AdaBoost\horseColicTest2.txt",
                      sep='\t',header=None,index_col=None)

x_train,y_train=extract_x_y(training_set)
x_test,y_test=extract_x_y(testing_set)

## training classifier
adaboostClassifier=adaBoostTrain(x_train, y_train, 1000)

## test the classifier
y_pred=adaClassify(x_test,adaboostClassifier)
test_errorRate=np.shape(y_test[y_pred!=y_test])[1]/np.shape(y_test)[0]

## assemble
plot_loss(x_train, y_train, 100, x_test, y_test)
