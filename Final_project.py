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
import re
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

#import os
#os.chdir('/Users/yuranpan/desktop/Fordham/Data_Mining/Project/Readmittedrate_master')

dataset_raw = pd.read_csv('diabetic_data.csv')
dataset_raw.dtypes
dataset_raw.describe(include = 'all')

### create a new to combine "> 30" and "no" to "no" ###
#getting the number of target
sns.set() # seaborn setting
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
numeric = np.array(dataset.describe().columns)

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
sns.set(style="ticks")
#scatters between numeric features and color by targets
sns.pairplot(dataset[numeric],palette='husl',hue = 'readmitted', plot_kws={'alpha': 0.5})



### plot heatmap of coeffiecient for numeric data###
sns.heatmap(dataset[numeric].corr()) 



#######################################
####### Handling Missing Values #######
#######################################
##drop
#1.drop features: weight, payercode and medical_specialty 
dataset_1 = dataset.drop(labels='weight',axis = 1).drop(labels='payer_code',axis = 1).drop(labels='medical_specialty',axis = 1)

#2.dropping the instances who have missing values in race,diag1,2,3
dataset_2 = dataset_1.dropna(subset=['race','diag_1','diag_2','diag_3'])

#3.droping the instances who are dead!
#Integer identifier corresponding to 29 distinct values, for example, discharged to home, expired, and not available
def drop_dischaged(data,col,values):
    for value in values:
        data = data.drop(data[(data[col] == value)].index)
    return data

dead_code = [11,12,13,14,19,20,21,25,26]#people are dead
dataset_3 = drop_dischaged(dataset_2,'discharge_disposition_id',dead_code)

##mapping
#4.transfer objects into numbers 
#ageing bins to 1:10
age_mapping = {'[0-10)':1,
               '[10-20)':2,
               '[20-30)':3,
               '[30-40)':4,
               '[40-50)':5,
               '[50-60)':6,
               '[60-70)':7,
               '[70-80)':8,
               '[80-90)':9,
               '[90-100)':10}
dataset_3['age'] = dataset_3.age.map(age_mapping)

#max_glu_serum: Indicates the range of the result or if the test was not taken.
#  Values: “>200,” “>300,” “normal,” and “none” if not measured
max_glu_serum_mapping = {'>200':2, '>300':3, 'None':0, 'Norm':1}
dataset_3['max_glu_serum'] = dataset_3.max_glu_serum.map(max_glu_serum_mapping)

#A1Cresult :Indicates the range of the result or if the test was not taken. Values: “>8” if the result was greater than 8%, 
# “>7” if the result was greater than 7% but less than 8%, “normal” if the result was less than 7%, and “none” if not measured.
A1Cresult_mapping = {'>7': 7, '>8':8, 'None':0, 'Norm':4}
dataset_3['A1Cresult'] = dataset_3.A1Cresult.map(A1Cresult_mapping)

#Change of medications: Indicates if there was a change in diabetic medications 
# (either dosage or generic name). Values: “change” and “no change”
change_mapping = {'Ch': 1, 'No': 0}
dataset_3['change'] = dataset_3.change.map(change_mapping)

#Diabetes medications : Indicates if there was any diabetic medication prescribed. 
# Values: “yes” and “no”
dia_mapping = {'No': 0, 'Yes': 1}
dataset_3['diabetesMed'] = dataset_3.diabetesMed.map(dia_mapping)

##One hot
#For medicine : Values: “up” if the dosage was increased during the encounter, “down” if the dosage was decreased, 
# “steady” if the dosage did not change, and “no” if the drug was not prescribed
data_medicine = dataset_3.iloc[:,21:44]
data_medicine = pd.get_dummies(data_medicine)

#for race and gender
dataset_race_gender = pd.get_dummies(dataset_3[['race','gender']])


##Diag1,2,3 mapping
#Diag1 
data_diag = dataset_3[['diag_1','diag_2','diag_3']]
#converting Vdd and Edd to 9999,floating number
data_diag['diag_1'] = data_diag['diag_1'].str.replace(r'[A-Z]\d+', '9999')
data_diag['diag_2'] = data_diag['diag_2'].str.replace(r'[A-Z]\d+', '9999')
data_diag['diag_3'] = data_diag['diag_3'].str.replace(r'[A-Z]\d+', '9999')

#
def create_list (first, last, extra_val = -999):
    if extra_val == -999:
        return list(range(first, last+1))
    else:
        other = list(range(first, last+1))
        print(other)
        other.append(extra_val)
        return other

data_diag.diag_2 = data_diag.diag_2.astype(float)
data_diag.diag_1 = data_diag.diag_1.astype(float)
data_diag.diag_3 = data_diag.diag_3.astype(float)
df = data_diag

list_of_unique_ints = pd.unique(df.values.ravel())

circ = create_list(390, 459, 758)
resp = create_list(460, 519, 786)
dige = create_list(520, 579, 787)
inj  = create_list(800, 999)
musc = create_list(710, 739)
geni = create_list(580, 629, 788)
neo =  create_list(140, 239)

df = df.astype(int)
df = df.replace(circ, 'Circulatory')
df = df.replace(resp, 'Respiratory')
df = df.replace(dige, 'Digestive')
df = df.replace(250, 'Diabetes')
df = df.replace(inj, 'Injury')
df = df.replace(musc, 'Musculoskeletal')
df = df.replace(geni ,'Genitourinary')
df = df.replace(neo, 'Neoplasms')
df = df.replace(list_of_unique_ints, 'Other')

data_diag_ = pd.get_dummies(df)



##Conbine all
data_id = dataset_3[['encounter_id','patient_nbr']]
data_meddle = dataset_3.iloc[:,4:15] #before diag 
data_numToA1 = dataset_3.iloc[:,18:21]
data_tail = dataset_3.iloc[:,-3:] # change and med and target
dataset_cleaned =pd.concat([data_id,dataset_race_gender,data_meddle,data_diag_,data_numToA1,data_medicine,data_tail],axis = 1) 
dataset_cleaned.to_csv('data_cleaned.csv')
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























