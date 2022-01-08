## part1 Lenses classify
import os 
os.chdir("G:/project/机器学习实战/My_Machine Learning in Action/chapter3 Decision-tree/")

import numpy as np
import pandas as pd
import math

#%% load data
dataset=pd.read_table('lenses.txt',sep='\t',header=None)
features_names=['age','prescript','astigmatic','tearRate'] ## features names, column names
#%% calculate Shannon Entropy 
def cal_entropy(dataSet):
    ## for a dataset, which the last column is label
    ## for calculating entropy, we need to calculate the properity first
    dataSet_size=np.shape(dataSet)[0] ## the instance number
    labels=dataSet.iloc[:,-1]
    
    labels=labels.value_counts()
    
    ## calculate properity
    label_properity=dict()
    for i in range(0,len(labels)):
        label_properity[labels.index[i]]=labels[i]/dataSet_size
        
    ## calculate entropy
    shannon_entropy=0
    for i in range(0,len(labels)):
        shannon_entropy+=label_properity[labels.index[i]]*(math.log2(label_properity[labels.index[i]]))
    shannon_entropy=0-shannon_entropy
    return(shannon_entropy)
        
#%% split the dataset via a features
def splitedataset(dataSet,axis,value):
    ## obtain the axis' label equal to value as sub-dataset
    residue_dataset=pd.DataFrame()
    for i in range(0,np.shape(dataSet)[0]):
        feature_label_tem=dataSet.iloc[i,axis]
        if(feature_label_tem==value):
            residue_dataset=residue_dataset.append(dataSet.iloc[i,:])
    if(np.shape(residue_dataset)[1]>2):
        del residue_dataset[axis]
    return residue_dataset

#%% choose the best feature
## use the entropy reduction to select features
def chooseBestFeatureToSplit(dataSet):
    features_number=np.shape(dataSet)[1]-1
    
    base_entropy=cal_entropy(dataSet)
    best_information_gain=0.0
    best_feature=-1
    
    for i in range(0,features_number):
        feature_labels=dataSet.iloc[:,i]
        feature_labels=feature_labels.value_counts().index
        
        new_entropy=0
        ## split dataSet and sum the entropy of each sub
        for j in range(0,len(feature_labels)):
            dataSubset=splitedataset(dataSet, i, feature_labels[j])
            data_Subset_size=np.shape(dataSubset)[0]
            
            ## each features will divide dataSets into more than one dataSubset, 
            ## use the prob to multiple the new entropy of each dataSubset
            prob=data_Subset_size/np.shape(dataSet)[0] 
            new_entropy+=prob*cal_entropy(dataSubset)
        
        ## calculate the information gain
        information_gain=new_entropy-base_entropy
        
        if(information_gain<best_information_gain):
            best_information_gain=information_gain
            best_feature=i
            
    return best_feature

#%% define the leaf node label 
## for each leaf-node, we need to define the final label to output
def defineLeafNodeLabel(dataSet):
    labels=dataSet.iloc[:,-1].value_counts()
    ## the label which is the frequenctly occuring will be chosen
    return labels.index[0]

#%% define the function of creat tree
def creatTree(dataSet,features_names):
    ## two terminate condition
    ## all labels in one data subset are identity
    labels=dataSet.iloc[:,-1]
    if(np.shape(labels.value_counts())[0]==1):
        return labels.value_counts().index[0]
    ## all features are consumed
    if(np.shape(dataSet)[1]==1):
        return defineLeafNodeLabel(dataSet)
    
    ## other condition
    bestfeture=chooseBestFeatureToSplit(dataSet)
    bestfeture_label=features_names[bestfeture]
    
    tree={bestfeture_label:{}}
    
    ## remove the features 
    residue_label=features_names[:]
    del residue_label[bestfeture]
    
    ## split to the data-subset
    best_feature_labels=dataSet.iloc[:,bestfeture].value_counts().index

    for i in range(len(best_feature_labels)):
        tree[bestfeture_label][best_feature_labels[i]]=creatTree(dataSet=splitedataset(
            dataSet=dataSet,axis=bestfeture,value=best_feature_labels[i]),
            features_names=residue_label)

    return tree

a=creatTree(dataSet=dataset, features_names=features_names)

#%% visualize the decision-tree
## I do not learn the code about this part, while the sklearn model contain the enough functions to access it.
from treePlotter import createPlot

createPlot(a) ## there are some wrongs, but make no sense
