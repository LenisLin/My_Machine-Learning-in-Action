## part1 dating-data
#%% load modules
import sys
sys.path.append("G:\anaconda\envs\python_basic\Lib\site-packages")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle

#%% data pre-processing
dataset=pd.read_csv('./Dating/datingTestSet.txt',sep='\t',header=None)
data_size=np.shape(dataset)[0]
dataset=shuffle(dataset)

## training set and testing set
training_size=(data_size//10)*9
testing_size=data_size-training_size

## 9:1
training_set=dataset.iloc[0:training_size,:]
testing_set=dataset.iloc[training_size:data_size,]
del dataset

## view the datasets
training_set.head()

# %% z-score normalization function
def z_score_normalize(array):
    _mean=np.mean(array)
    _std=np.std(array)
    return _mean,_std

# %% KNN main function
def KNN(Dataset,inputX,K):
# %%% LabelEncoder
    ## extract label of Datasets
    label=Dataset.iloc[:,-1]
    label_encoder=LabelEncoder()
    label_encoder=label_encoder.fit(label)
    training_label=label_encoder.transform(label) ## Transform Categories Into Integers 
    #print(label_encoder.inverse_transform(training_label)) ## Transform Integers Into Categories
    del label
    training_data=Dataset.iloc[:,0:(np.shape(Dataset)[1]-1)]
    
# %%% normalize
    for i in range(0,(np.shape(training_data)[1])):
        tem_mean,tem_std=z_score_normalize(training_data.iloc[:,i])
        training_data.iloc[:,i]=(training_data.iloc[:,i]-tem_mean)/tem_std
        inputX[i]=(inputX[i]-tem_mean)/tem_std
    
    del tem_mean,tem_std,i
# %%%  calculate distance
    ## the distance is L2(Eucliden) distance
    inputX_multi=np.tile(inputX, reps=(np.shape(training_data)[0],1)) ## numpy.tile(A,rep(n,m)) ## https://blog.csdn.net/yeshang_lady/article/details/107286350
    
    ## as numpyt matrix form
    inputX_multi=np.matrix(inputX_multi)
    training_data=np.matrix(training_data)
    del inputX
    
    ## Eucliden distance
    diff=training_data-inputX_multi
    diff=np.power(diff,2)
    
    diff=diff.sum(axis=1)
    
#%%% predict the label
    diff_df=pd.DataFrame()
    diff_df["label"]=training_label
    diff_df["distance"]=np.array(diff)
    del diff,inputX_multi
    
    diff_df=diff_df.sort_values(by="distance",ascending=True)
    
    ## select the k as the voters
    diff_df=diff_df.iloc[0:K,]
    voter=diff_df["label"].value_counts()
    
    return label_encoder.inverse_transform([voter.index[0]])[0]

# %% testing the accuracy

true_label=testing_set.iloc[:,-1]
predict_label=[]
for i in range(0,testing_size):
    testing_data=testing_set.iloc[i,0:3]
    predict_label.append(KNN(Dataset=training_set, inputX=testing_data, K=6))    
del i
true_number=0
true_label.index=range(0,testing_size)
for i in range(0,len(true_label)):
    if(true_label[i]==predict_label[i]):
        true_number+=1
