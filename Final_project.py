#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:08:18 2018

@author: yuranpan
"""

import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn import linear_model
from sklearn import svm
from sklearn.neighbors import NearestNeighbors
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.feature_selection import VarianceThreshold
from sklearn import preprocessing
import matplotlib.pyplot as plt


dataset_raw = pd.read_csv('diabetic_data.csv')
dataset_raw.dtypes
dataset_raw.describe(include = 'all')

### create a new to combine "> 30" and "no" to "no" ###

count_by_readmitted = dataset_raw.groupby('readmitted').size()
print(count_by_readmitted)
count_by_readmitted.plot(kind = 'bar')# inbalanced data


### convert '?' to 'NaN' ###
dataset = dataset_raw.replace('?', np.nan)    

#id and phone number are not factor to the target, convert to str
dataset.encounter_id = dataset.encounter_id.astype('str')
dataset.patient_nbr = dataset.patient_nbr.astype('str')
dataset.info() # check again


########################################
##### Exploring Categorical Variables ##
########################################

n = dataset.shape[0]
features_all= dataset.columns
'''features_all =
        ['encounter_id', 'patient_nbr', 'race', 'gender', 'age', 'weight',
       'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
       'time_in_hospital', 'payer_code', 'medical_specialty',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient', 'diag_1',
       'diag_2', 'diag_3', 'number_diagnoses', 'max_glu_serum', 'A1Cresult',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted']'''

### check percentage of missing values ###
df_feature_catg = dataset.select_dtypes(include= [object])
feature_catg = df_feature_catg.columns # object features
d_catg = df_feature_catg.shape[1]

#Missing ratio for each feature
missing = dataset.count()/dataset.shape[0]
#Sorting accending 
missing.sort_values()

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
#Plotting numeric hist
numeric = ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
       'time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses']

for i in numeric:
    dataset[i].plot(kind ='hist')
    plt.xlabel(i)
    plt.show()

##########################################

### find percentage of missing values in each feature ###

# try df.isnull().sum()
dataset.isnull().sum() 



### plot scatterplot matrix ###
import seaborn as sns
sns.set(style="ticks")
sns.pairplot(dataset[numeric])




### plot heatmap of coeffiecient ###
sns.heatmap(dataset[numeric].corr()) 



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


