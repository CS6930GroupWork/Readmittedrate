#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 11:08:18 2018

@author: 
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
from pandas.plotting import scatter_matrix

#import os
#os.chdir('/Users/yuranpan/desktop/Fordham/Data_Mining/Project/Readmittedrate_master')

dataset_raw = pd.read_csv('diabetic_data.csv')
dataset_raw.dtypes
dataset_raw.describe(include = 'all')

### create a new to combine "> 30" and "no" to "no" ###
#getting the number of target
count_by_readmitted = dataset_raw.groupby('readmitted').size()
print(count_by_readmitted)
count_by_readmitted.plot(kind = 'bar')# inbalanced data


### convert '?' to 'NaN' and make a dataframe copy dataset,all work we done should on this set ###
dataset = dataset_raw.replace('?', np.nan).copy()   

#making the target to be 0,1: 0 means no readmitted, 1 means readmitted.
dataset.readmitted.replace('<30',1,inplace= True)  
dataset.readmitted.replace('NO',0, inplace= True)  
dataset.readmitted.replace('>30',0, inplace= True) 




#id and phone number are not factor to the target, convert to str
dataset.encounter_id = dataset.encounter_id.astype('str')
dataset.patient_nbr = dataset.patient_nbr.astype('str')
dataset.info() # check again

# separate features and label
dataset_X = dataset.iloc[:,0:-1]
dataset_Y = dataset.iloc[:,-1]


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

# df.isnull().sum(),and How many percerage of missings values we have
dataset_pct_missing = dataset.isnull().sum()/n
dataset_pct_missing.sort_values(ascending=[False])

''' 
insights: there are 7 catg feature have missing values, these are
['weight', 'medical_specialty', 'payer_code', 'race', 'diag_3', 'diag_2','diag_1'
insights: Weight has 96.86% of missing values

'''

### Histogram ###

##########################################
#Plotting numeric hist
numeric = np.array(dataset.select_dtypes(include= [int64]).columns)


hist_matrix = dataset[numeric].hist()
plt.show()

#distributions graphes for numeric features:
dist_matrix = dataset.plot(kind='density', subplots=True, layout=(4,4), sharex=False)
plt.show()

#boxplot graphes for numeric features:
boxplot_matrix = dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
plt.show()
##########################################

### find percentage of missing values in each feature ###





### plot scatterplot matrix for numeric data ###
import seaborn as sns
sns.set(style="ticks")
sns.pairplot(dataset[numeric])

# comparing each features with the target
for feature in numeric:
    g = sns.JointGrid(data=dataset, x=feature, y='readmitted') 
    g = g.plot_joint(sns.kdeplot)
    g = g.plot_marginals(sns.kdeplot, shade=True)
    g = g.annotate(stats.pearsonr)

### plot heatmap of coeffiecient for numeric data###
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


