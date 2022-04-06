# -*- coding: utf-8 -*-
## functions for adaboost
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
## test data
def load_Simple_Data():
    datMat=np.matrix([[1,2.1],
                   [2,1.1],
                   [1.3,1],
                   [1,1],
                   [2,1]])
    
    classLabel=np.matrix([[1],[1],[-1],[-1],[1]])
    
    return datMat,classLabel

## using stump to classify 
def stumpClassify(data,classify_feature,thresh_value,thresh_sign):
    return_label=np.ones((np.shape(data)[0],1)) ## ones(row_len,col_len)
    if thresh_sign=='lt':
        return_label[data[:,classify_feature]>=thresh_value]=(-1)
    else:
        return_label[data[:,classify_feature]<thresh_value]=(-1)
    return return_label

## build a stump 
def buildStump(x,y,D,step_number=10):
    m,n=np.shape(x)
    min_error=float("inf")
        
    num_step=step_number
    bestStump={}
    bestClasEst=np.matrix(np.zeros((m,1)))
    
    for features in range(0,n):
        step_start=float(min(x[:,features]))
        step_end=float(max(x[:,features]))
        step_length=(step_end-step_start)/num_step
        
        thresh_value=step_start-step_length
        while (True):
            thresh_value=thresh_value+step_length
            
            for thresh_sign in ["lt","gt"]:
                y_pred=stumpClassify(x,features,thresh_value,thresh_sign)

                correct_mat=np.ones(shape=(m,1))
                correct_mat[y_pred==y]=0
                
                wight_error=np.dot(np.transpose(D),correct_mat) 
                
                'print("split dim: %d, thresh value: %0.2f, thresh sign: %s, weight error: %.3f" % \
                      (features,thresh_value,thresh_sign,wight_error))'
                
                if (wight_error<min_error):
                    min_error=wight_error
                    bestClasEst=y_pred
                    bestStump['dim']=features
                    bestStump['thresh_value']=thresh_value
                    bestStump['thresh_sign']=thresh_sign
            if (thresh_value>=step_end):
                break;
    return bestStump,min_error,bestClasEst

## Training adaboost classifier based on stump classifier
def adaBoostTrain(x,y,interation_num=50):
    weakClassifiers=[]
    m,n=np.shape(x)
    D=np.ones(shape=(m,1))/m
    aggClassEst=np.zeros(shape=(m,1))
    
    for i in range(interation_num):
        bestStump,error,y_pred=buildStump(x, y, D, 10)
        print("D: ",np.transpose(D))
        alpha=math.log((1-error)/error)/2
        bestStump["alpha"]=alpha
        weakClassifiers.append(bestStump)
        print("classEst: \n",y_pred.T)
        
        D_sum=np.sum(D)
        D[y_pred==y]=D[y_pred==y]*math.exp(-alpha)
        D[y_pred!=y]=D[y_pred!=y]*math.exp(alpha)
        D=D/D_sum
        
        aggClassEst=aggClassEst+y_pred*alpha ## all classifier to classify a sample
        print("aggClassEst: ",np.sign(np.transpose(aggClassEst)))
        aggErrors=np.sum(np.sign(aggClassEst)!=np.mat(y))
        aggErrors=np.multiply(np.sign(aggClassEst)!=y,np.ones((m,1)))
        errorRate=np.sum(aggErrors)/m
        print("total error rate: ",errorRate)
        bestStump["errorRate"]=errorRate
        if(errorRate==0):
            break;
    return weakClassifiers

## test adaboost classifier
def adaClassify(data,classifier):
    data=np.mat(data)
    m,n=np.shape(data)
    y_pred=np.zeros(shape=(m,1))
    
    for i in range(len(classifier)):
        ClassEst=stumpClassify(data, classifier[i]['dim'],
                               classifier[i]['thresh_value'] ,classifier[i]['thresh_sign'])
        y_pred=y_pred+ClassEst*classifier[i]['alpha']
    
    print("predict y: ",np.sign(np.transpose(y_pred)))
    return np.sign(y_pred)

## extract x and y from a dataset
def extract_x_y(dataset):
    m,n=np.shape(dataset)

    ## extract label
    y_train=dataset.iloc[:,n-1]
    y_train.head()
    y_train=np.mat(y_train)
    y_train=y_train.T

    ## extract features
    x_train=dataset.iloc[:,0:n-1]
    x_train=np.mat(x_train)
    m,n=np.shape(x_train)
    
    print("x dims: ",np.shape(x_train)," y dims: ",np.shape(y_train))
    return x_train,y_train

## plot the loss of classifier
def plot_loss(classifier):
    plot_df=pd.DataFrame()
    plot_df["iteration"]=range(1,(len(classifier)+1))
    plot_df["loss"]=[classifier[i]["errorRate"] for i in range(len(classifier))]
    
    plt.plot(plot_df["iteration"], plot_df["loss"], 'r-', label=u'Loss')
    plt.show()

## plot the loss of training and testing
def plot_loss(x_train, y_train, iteration_num, x_test, y_test):
    plot_df=pd.DataFrame(data=None)

    for i in range(1,iteration_num+1):
        adaboostClassifier=adaBoostTrain(x_train, y_train, i)
        y_pred=adaClassify(x_test,adaboostClassifier)
        test_errorRate=np.shape(y_test[y_pred!=y_test])[1]/np.shape(y_test)[0]
        df_tem=np.mat([[i,adaboostClassifier[i-1]["errorRate"],test_errorRate]])
        plot_df=plot_df.append(pd.DataFrame(df_tem))
    
    plot_df.columns=["iteration","train_ErrorRate","test_ErrorRate"]
    
    plt.plot(plot_df["iteration"], plot_df["train_ErrorRate"], 'r-', label=u'Training loss')
    plt.plot(plot_df["iteration"], plot_df["test_ErrorRate"], 'b-', label=u'Testing loss')
    plt.show()
    return None