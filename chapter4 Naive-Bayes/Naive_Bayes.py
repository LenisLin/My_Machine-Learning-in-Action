# -*- coding: utf-8 -*-
## Naive_Bayes for spam recognization
import os
import re
import pandas as pd
import numpy as np
#%% load datasets
def loadDatas(path):
    examples=os.listdir(path)
    returnlist=list()
    for i in range(0,len(examples)):
        file_tem=open(path+examples[i],encoding='windows-1252')
        allwords=file_tem.readlines()
        a_file_words=[]
        for j in range(len(allwords)):
            word_tem=allwords[j]
            a=re.findall(pattern='\w+', string=word_tem)
            a_file_words.extend(a)
        file_tem.close()
        returnlist.append(a_file_words)
    return returnlist

hams_example=loadDatas(".\email\ham\\")
spams_example=loadDatas(".\email\spam\\")

## label
hams_label=["ham"]*len(hams_example)
spams_label=["spam"]*len(spams_example)
#%% form bag-of-words set
## this bag of words contains all the words in all examples
bagOfWords=set()
for i in hams_example:
    bagOfWords = bagOfWords | set(i)
for i in spams_example:
    bagOfWords = bagOfWords | set(i)
del i
bagOfWords=list(bagOfWords)
#%% form word-vector
def formWordsVector(bagOfWords,example_list):
    example_wordVector=pd.DataFrame(data=None,columns=bagOfWords)
    for i in range(0,len(example_list)):
        example_features=list()
        for j in range(0,len(bagOfWords)):
            if(bagOfWords[j] in example_list[i]):
                example_features.append(1)
            else:
                example_features.append(0)
        example_wordVector.loc[i,]=example_features
    return example_wordVector
    
hams_example_wordVector=formWordsVector(bagOfWords,hams_example)
hams_example_wordVector["label"]=hams_label
del hams_label,hams_example

spams_example_wordVector=formWordsVector(bagOfWords,spams_example)
spams_example_wordVector["label"]=spams_label
del spams_label,spams_example

#%% splite training set and test set
all_dataset=hams_example_wordVector.append(spams_example_wordVector)
del hams_example_wordVector,spams_example_wordVector

all_index=[i for i in range(0,np.shape(all_dataset)[0])]
np.random.shuffle(all_index) ## no return 
trainingset_index=all_index[0:45]
testingset_index=all_index[45:50]

training_set=all_dataset.iloc[trainingset_index,:]
testing_set=all_dataset.iloc[testingset_index,:]

del all_index,all_dataset,trainingset_index,testingset_index
#%% calculate condition properity
def trainNaiveBayesClassifier(TrainingSet):
    ## in this function, we need to calculate the condition properity for Bayes formula
    ## split training set with label
    labels=TrainingSet.iloc[:,(np.shape(TrainingSet)[1]-1)]
    labels=list(labels)
    
    TrainingSet=TrainingSet.iloc[:,0:(np.shape(TrainingSet)[1]-1)]+1
    
    spams_index=[i for i in range(0,len(labels)) if labels[i]=="spam"]
    hams_index=[i for i in range(0,len(labels)) if labels[i]=="ham"]
    
    ## this is a two-class classifier, so we just need to calculate the p(c_0| w and  c_1|w)
    ## here we calculate p(c_0 | w), 0 represent the hams
    p_c0=len(hams_index)/(len(spams_index)+len(hams_index))

    ## calculate p(w|c0), which equal to π(i from 0 to n)(wi|c0)
    hams_data=TrainingSet.iloc[hams_index,:]
    hams_data=hams_data.iloc[:,0:(np.shape(hams_data)[1]-1)]
    hams_data_sum=np.sum(hams_data,axis=0)
    p_wi_c0=hams_data_sum/len(hams_index)
    
    p_w_c0=1
    for i in range(len(p_wi_c0)):
        p_w_c0=p_w_c0*p_wi_c0[i]
    
    del i
    
    ## here we calculate p(c_1 | w), 1 represent the spams
    p_c1=len(spams_index)/(len(spams_index)+len(hams_index))

    ## calculate p(w|c0), which equal to π(i from 0 to n)(wi|c0)
    spams_data=TrainingSet.iloc[spams_index,:]
    spams_data=spams_data.iloc[:,0:(np.shape(spams_data)[1]-1)]
    spams_data_sum=np.sum(spams_data,axis=0)
    p_wi_c1=spams_data_sum/len(spams_index)
    
    p_w_c1=1
    for i in range(len(p_wi_c1)):
        p_w_c1=p_w_c1*p_wi_c1[i]
    
    del i,labels,spams_data,spams_data_sum,hams_data,hams_data_sum,hams_index,spams_index
    
    ## calculate p(w), which equals to π(from 0 to n)p(wi)
    TrainingSet=TrainingSet.iloc[:,0:(np.shape(TrainingSet)[1]-1)]
    TrainingSet_sum=np.sum(TrainingSet,axis=0)
    p_wi=TrainingSet_sum/len(TrainingSet_sum)
    
    ## calculate p(w), which equals to π(from 0 to n)p(wi)
    TrainingSet=TrainingSet.iloc[:,0:(np.shape(TrainingSet)[1]-1)]
    TrainingSet_sum=np.sum(TrainingSet,axis=0)
    p_wi=TrainingSet_sum/len(TrainingSet_sum)
    
    ## calculate p(c0|w) vector
    c0_vector=p_wi_c0*p_c0/p_wi
    
    ## calculate p(c0|w) vector
    c1_vector=p_wi_c1*p_c1/p_wi
    
    c0_vector=c0_vector.dropna()
    c1_vector=c1_vector.dropna()

    return c0_vector,c1_vector

c0_vector,c1_vector = trainNaiveBayesClassifier(training_set)

def predict_label(c0_vector,c1_vector,word_vector):
    word_feature=set(word_vector.index)
    model_feature=set(c0_vector.index)

    intersect_features=word_feature & model_feature
    del word_feature,model_feature
    
    c0_vector=c0_vector[intersect_features]
    c1_vector=c1_vector[intersect_features]
    word_vector=word_vector[intersect_features]
    
    c0_score=np.transpose(c0_vector)*word_vector
    c0_score=c0_score.sum(axis=0)
    
    c1_score=np.transpose(c1_vector)*word_vector
    c1_score=c1_score.sum(axis=0)
    
    if(c0_score>c1_score):
        return ("hams")
    else:
        return ("spams")

testing_set_labels=testing_set.iloc[:,(np.shape(testing_set)[1]-1)]
test_predict_label=[]
for i in range(np.shape(testing_set)[0]):
    vector_tem=testing_set.iloc[i,0:(np.shape(testing_set)[1]-1)]
    label_tem=predict_label(c0_vector,c1_vector,vector_tem)
    test_predict_label.append(label_tem)
del vector_tem,label_tem
