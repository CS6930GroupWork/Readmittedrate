#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:08:18 2018

@author: yuranpan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pylab 
import scipy.stats as stats
import os
from sklearn.model_selection import train_test_split
%matplotlib inline

os.chdir('/Users/yuranpan/Desktop/Fordham/Data_Mining/Project')

dataset_raw = pd.read_csv('diabetic_data.csv')
dataset_raw.dtypes
dataset_raw.describe(include = 'all')

### create a new to combine "> 30" and "no" to "no" ###

a = dataset.groupby('readmitted').size()
dataset['readmitted_new'] = np.where(dataset_raw['readmitted'] == '<30','<30','NO')
dataset['readmitted_new'].hist()  # inbalanced data

### convert '?' to 'NaN' ###
dataset = dataset.replace('?','')    



########################################
##### Exploring Categorical Variables ##
########################################

n = dataset.shape[0]
features_all= dataset.columns

### check percentage of missing values ###
df_feature_catg = dataset.select_dtypes(include= [object])
feature_catg = df_feature_catg.columns
d_catg = df_feature_catg.shape[1]

dataset.groupby('race').size()
dataset['race'].isnull().sum()  # it couldn't detect the NaN

pct_nan_list = []
for feature in feature_catg:
    pct_nan = (dataset.groupby(feature).size().get('NaN',0))/n
    print('feature',feature,'has',pct_nan,'of missing values')
    pct_nan_list.append(pct_nan)
num_feature_missing = np.count_nonzero(pct_nan_list)       
index = argsort(pct_nan_list)[::-1][0:7]
feature_catg = feature_catg[index]
print(pct_nan_list)
print('top 7 feature with most missing value is',feature_catg)

''' 
insights: there are 7 catg feature have missing values, these are
['weight', 'medical_specialty', 'payer_code', 'race', 'diag_3', 'diag_2','diag_1'
insights: Weight has 96.86% of missing values

'''

### Histogram ###

#hist = df_feature_catg.hist() # code not working

fig = plt.figure()
for i in range(d_catg):
    axe = fig.add_subplot(d_catg/3, 3,i+1)
    axe.hist(df_feature_catg.iloc[:,i])

### Q-Q plot ###

#stats.probplot(df_feature_catg, dist="norm", plot=pylab)
#pylab.show()


##########################################
######## Exploring Numerical Variables ###
##########################################

### find percentage of missing values in each feature ###

# try df.isnull().sum()




### plot scatterplot matrix ###





### plot heatmap of coeffiecient ###



#######################################
####### Handling Missing Values #######
#######################################


'''
FOR NUMERICAL FEATURES:
    should we drop them?  df.dropna()
    should we fill in them with Mean, Median, Mode ?
    should we use predictive models?


FOR CATEGORICAL FEATURES:
    ordinal or normial?
    convert values to factors:
        get_dummies
        from sklearn.preprocessing import LabelEncoder
https://www.kaggle.com/danavg/dummy-variables-vs-label-encoding-approach 
    
    
'''
























### split the dataset ###
dataset_train, dataset_test = train_test_split(dataset, test_size = 0.2,train_size = 0.8, random_state = 42,
                                               shuffle = True)


