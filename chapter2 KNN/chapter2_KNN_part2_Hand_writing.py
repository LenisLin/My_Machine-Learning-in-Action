## part2 hand-writing identify
#%% load modules
import sys
sys.path.append("G:\anaconda\envs\python_basic\Lib\site-packages")

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from chapter2_KNN_part1_Dating import KNN
import os

#%% load datasets and extract label
print(os.getcwd())

training_file_names=os.listdir("./digits/trainingDigits")
training_label=[]
for i in range(0,len(training_file_names)):
    label_tem=training_file_names[i].split('_')[0]
    label_tem=int(label_tem)
    training_label.append(label_tem)
del label_tem,i

os.chdir("./digits/trainingDigits")
file_tem=open(training_file_names[0], "r")
file_tem.read
