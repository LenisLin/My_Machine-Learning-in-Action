# -*- coding: utf-8 -*-
## Logistic regression for horse survival prediction
#%% import module
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
#%% load datasets
training_set=pd.read_table(filepath_or_buffer="./horseColicTraining.txt",header=None,sep='\t')
training_set.head()

test_set=pd.read_table(filepath_or_buffer="./horseColicTest.txt",header=None,sep='\t')
test_set.head()

#%% pre-process the dataset
## NA processing
print(training_set.isnull().values.any()) 
print(test_set.isnull().values.any()) 

## extract labels
train_x=training_set.iloc[:,0:(np.shape(training_set)[1]-1)]
train_y=training_set.iloc[:,np.shape(training_set)[1]-1]

test_x=test_set.iloc[:,0:(np.shape(test_set)[1]-1)]
test_y=test_set.iloc[:,np.shape(test_set)[1]-1]
del [training_set,test_set]

## normalization and scaling
for i in range(0,np.shape(train_x)[1]):
    sigma_=np.std(train_x.iloc[:,i])
    mean_=np.mean(train_x.iloc[:,i])
    train_x.iloc[:,i]=(train_x.iloc[:,i]-mean_)/sigma_

for i in range(0,np.shape(test_x)[1]):
    sigma_=np.std(test_x.iloc[:,i])
    mean_=np.mean(test_x.iloc[:,i])
    test_x.iloc[:,i]=(test_x.iloc[:,i]-mean_)/sigma_
del sigma_,mean_,i
#%% using pytorch to realize logistic-regression
## definde a network
class Logisticregression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear=nn.Linear(21, 2)
    
    def forward(self,x):
        out=self.linear(x)
        out=torch.sigmoid(out)
        return out
#%% test function
def test(pred,lab):
    t=pred.max(-1)[1]==lab ## 每个实例预测最大的预测概率
    return torch.mean(t.float())
#%% some hyperparameters
net=Logisticregression()
criterion=nn.CrossEntropyLoss()
optimization=torch.optim.Adam(net.parameters())
epochs=1000
#%% training 
for epoch in range(epochs):
    net.train()
    
    x=torch.from_numpy(np.array(train_x)).float()
    y=torch.from_numpy(np.array(train_y)).long()
    
    y_output=net(x)
    loss=criterion(y_output,y)
    optimization.zero_grad()
    loss.backward()
    optimization.step()
    if (epoch+1)%100 ==0 : # 这里我们每100次输出相关的信息
        # 指定模型为计算模式
        net.eval()
        test_in=torch.from_numpy(np.array(test_x)).float()
        test_l=torch.from_numpy(np.array(test_y)).long()
        test_out=net(test_in)
        # 使用我们的测试函数计算准确率
        accu=test(test_out,test_l)
        print("Epoch:{},Loss:{:.4f},Accuracy：{:.2f}".format(epoch+1,loss.item(),accu))
