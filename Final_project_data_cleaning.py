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
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from scipy import stats

import os
#os.chdir('E:\\Fordham\\2018 Fall\\CISC 6930\\Final Project')
os.chdir('/Users/yuranpan/Desktop/Fordham/Data_Mining/Project/Readmittedrate-master')

dataset_raw = pd.read_csv('diabetic_data.csv')
#*** screen shot the graph
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


sns.set()
dataset['readmitted'].replace('<30',1, inplace = True)
dataset['readmitted'].replace('NO',0, inplace = True)
dataset['readmitted'].replace('>30',0, inplace = True)
#*** print the graph
plt.hist(dataset['readmitted'])
plt.xlabel('readmitted')


### fix feature types ####

# numeric to nominal
dataset.encounter_id = dataset.encounter_id.astype('object')
dataset.patient_nbr = dataset.patient_nbr.astype('object')

# convert numeric code to category code
dataset['admission_type_id'] = dataset['admission_type_id'].astype('str')
dataset['discharge_disposition_id'] = dataset['discharge_disposition_id'].astype('str')
dataset['admission_source_id'] = dataset['admission_source_id'].astype('str')

#*** changed the datatype accord to feature meaning
dataset.info() # check again



########################################
##### Exploring Categorical Variables ##
########################################

n = dataset.shape[0]
features_all= dataset.columns
'''
  features_all =
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
       'metformin-pioglitazone', 'change', 'diabetesMed', 'readmitted'],
      dtype='object']
 '''
 

### check percentage of missing values ###
df_feature_catg = dataset.select_dtypes(exclude= ['int64'])
feature_catg = df_feature_catg.columns # object features
d_catg = df_feature_catg.shape[1]

# df.isnull().sum(),and How many percerage of missings values we have
dataset_pct_missing = dataset.isnull().sum()/n
#** print screen shot
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

'''
'time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses','readmitted'(shouldn'be)
'''
sns.set()
hist_matrix = dataset[numeric].hist()
#dataset['number_outpatient'].plot(kind='hist')


#distributions graphes for numeric features:
dist_matrix = dataset[numeric].plot(kind='density', subplots=True, layout=(2,4), sharex=False, sharey = False)
plt.show()    # No graphs showing

#boxplot graphes for numeric features:
sns.set()
boxplot_matrix = dataset.plot(kind='box', subplots=True, layout=(4,4), sharex=False, sharey=False)
plt.show()
##########################################




### plot scatterplot matrix for numeric data ###
sns.set(style="ticks")
#scatters between numeric features and color by targets
sns.pairplot(dataset[numeric],palette='husl',hue = 'readmitted', plot_kws={'alpha': 0.5})


### plot heatmap of coeffiecient for numeric data###
sns.heatmap(dataset[numeric].corr()) 

### crosstable
pd.crosstab(dataset['payer_code'],dataset['readmitted'],normalize = 'index')
pd.crosstab(dataset['age'],dataset['readmitted'],normalize = 'index')
pd.crosstab(dataset['age'],dataset['readmitted'],normalize = 'columns')
pd.crosstab(dataset['discharge_disposition_id'],dataset['readmitted'], normalize = 'index')
pd.crosstab(dataset['medical_specialty'],dataset['age'],normalize = 'index')


pd.crosstab([dataset['race'],dataset['age'],dataset['gender']],dataset['readmitted'],normalize = 'index')
pd.crosstab([dataset['race'],dataset['gender'],dataset['age']],dataset['readmitted'],normalize = 'index').to_csv('asian.csv')
dataset.groupby(['race','gender','age']).size()
pd.crosstab(dataset['race'],dataset['readmitted'],normalize = 'index')
pd.crosstab(dataset['gender'],dataset['readmitted'],normalize = 'index')
#######################################
####### Handling Missing Values #######
#######################################


# drop weights, very sparse 
dataset = dataset.drop(labels='weight',axis = 1)
# drop observation that has gender of unknown value
dataset = dataset[dataset['age']!= 'Unknown/Invalid']


# explore payercode, calcuting readmissoin distribution in each of the group 
payercode_bar = sns.countplot(x = 'payer_code', hue = 'readmitted', data = dataset)
# seems like payercode is not a signficant factor that affect readmisson 
#drop payercode
dataset = dataset.drop(labels='payer_code',axis = 1)



# assign medical specialty with mode based on age 

count_by_medspe = dataset.groupby('medical_specialty').size()
sns.countplot(x = 'age',hue = 'medical_specialty', data = dataset)


def conditional_fillna_base_on_mode(conditioned_feature,missing_value_feature, dataset):
    pair = {}
    age_grouped = dataset.groupby(conditioned_feature)
    for age, group in age_grouped:
        medical_grouped = group.groupby([missing_value_feature]).size().sort_values(ascending=False)  
        mode = medical_grouped.keys()[0]
        pair[age] = mode
    dataset[missing_value_feature] = dataset[missing_value_feature].replace(np.nan, pair[age])
    return dataset

dataset = conditional_fillna_base_on_mode('age','medical_specialty',dataset)



# race: fill in missing data with the mode
#** show table
dataset['race'].replace(np.nan,'Caucasian', inplace = True)

# drop the duplicated encounter_ID. Keep the last entry(Needs to check)
#dataset.drop_duplicates(subset = ['patient_nbr'], keep = 'last', inplace = True)
# decided to keep both records if duplicated. will make a feature to flag it.
# show how many duplicates there are for each patient_nbr


# drop paitence death
dead_code = ['11','12','13','14','19','20','21','25','26']
drop_discharged_index = dataset[dataset['discharge_disposition_id'].isin(dead_code)].index
dataset.drop(drop_discharged_index,inplace = True)                                       
#101766 - 3415 = 98351



#2.dropping the instances who have missing values in diag1,2,3
dataset.dropna(axis = 0, subset=['diag_1','diag_2','diag_3'], inplace = True)
#98351- 1485 = 96866



    





###############################
######## Encoding #############
###############################

#####Merge columns

# Mapping 
dataset['admission_source_id'].replace(['1','2'], 'Admitted because of physician/clinic referral', inplace = True)
dataset['admission_source_id'].replace('7', 'Admitted from emergency room', inplace = True)
# Merging
dataset['admission_source_id'][~dataset['admission_source_id'].isin(['Admitted because of physician/clinic referral',
        'Admitted from emergency room'])] = 'Others'
# check all the unique levels of admission source
dataset['admission_source_id'].unique()
# admission_source_id
#Admitted because of physician/clinic referral    29460
#Admitted from emergency room                     54853
#Others                                           12553


# Mapping
dataset['discharge_disposition_id'].replace('1', 'Discharged to home', inplace = True)
dataset['discharge_disposition_id'][~dataset['discharge_disposition_id'].isin(['Discharged to home'])] = 'Others'
# check again
dataset['discharge_disposition_id'].unique()
#Discharged to home    59005
#Others                37861
# Combine every sub-categories of surgery into 'Surgery'
dataset['medical_specialty'][dataset['medical_specialty'].isin(['Surgery-General',
        'Surgery-Neuro','Surgery-Cardiovascular/Thoracic',
        'Surgery-Colon&Rectal','Surgery-Plastic','Surgery-Thoracic',
        'Surgery-PlasticwithinHeadandNeck','Surgery-Pediatric',
        'Surgery-Maxillofacial','Surgery-Vascular','Surgery-Cardiovascular'])] = 'Surgery'

dataset['medical_specialty'][~dataset['medical_specialty'].isin(['InternalMedicine',
        'Family/GeneralPractice', 'Cardiology', 'Surgery'])] = 'Others'
# Check again
#dataset['medical_specialty'].unique()
#Cardiology                 5129
#Family/GeneralPractice     6953
#InternalMedicine          61765
#Others                    18279
#Surgery                    4740
'''
#####Label Encoding
#medical-specific features
labelencodinglist = ['metformin',
                     'repaglinide','nateglinide','chlorpropamide','glimepiride',
                     'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
                     'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide',
                     'citoglipton','insulin','glyburide-metformin','glipizide-metformin',
                     'glimepiride-pioglitazone','metformin-pioglitazone', 'metformin-rosiglitazone']

labelencoder = LabelEncoder()
for feature in labelencodinglist:
    labelencoder.fit(['No','Down','Steady','Up']).classes_
    dataset[feature] = labelencoder.transform(dataset[feature])

label encoding is based on alphbetical order. if the level matters, switch to ordinal encoding


#####Label Encoding
#medical-specific features
ordinalencodinglist = ['metformin',
                     'repaglinide','nateglinide','chlorpropamide','glimepiride',
                     'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
                     'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide',
                     'citoglipton','insulin','glyburide-metformin','glipizide-metformin',
                     'glimepiride-pioglitazone','metformin-pioglitazone', 'metformin-rosiglitazone']

ordinalencoder = OrdinalEncoder()

for feature in ordinalencodinglist:
    X = [['No', 0], ['Down', 1], ['Steady', 2],['Up',3]]
    ordinalencoder.fit(X)
    dataset[feature] = ordinalencoder.transform(dataset[feature].values.reshape(-1,1))
it seems not working
'''

########  Mapping and replace
encodinglist = ['metformin',
                     'repaglinide','nateglinide','chlorpropamide','glimepiride',
                     'acetohexamide','glipizide','glyburide','tolbutamide','pioglitazone',
                     'rosiglitazone','acarbose','miglitol','troglitazone','tolazamide','examide',
                     'citoglipton','insulin','glyburide-metformin','glipizide-metformin',
                     'glimepiride-pioglitazone','metformin-pioglitazone', 'metformin-rosiglitazone']


for feature in encodinglist: 
    level_mapper = {'No':0, 'Down':1,'Steady':2,'Up':3}
    dataset[feature] = dataset[feature].replace(level_mapper)




#age
# need to merge them into three groups <30, 30-60, 60-100
dataset['age'].replace(['[0-10)','[10-20)', '[20-30)'],'<30', inplace = True)
dataset['age'].replace(['[30-40)','[40-50)', '[50-60)'],'30-60', inplace = True)
dataset['age'].replace(['[60-70)', '[70-80)','[80-90)', '[90-100)'],'60-100', inplace = True)    


age_enc = dataset['age'].unique()
'''
labelencoder.fit(age_enc)
dataset['age'] = labelencoder.transform(dataset['age'])
'''
age_mapper = {'<30':0,'30-60':1,'60-100':2}
dataset['age'] = dataset['age'].replace(age_mapper)


#max_glu_serum
max_glu_serum_enc = dataset['max_glu_serum'].unique()
'''
labelencoder.fit(['None','Norm','>200','>300'])
dataset['max_glu_serum'] = labelencoder.transform(dataset['max_glu_serum'])
'''
maxgluserum_mapper = {'None':0,'Norm':1,'>200':2,'>300':3}
dataset['max_glu_serum'] = dataset['max_glu_serum'].replace(maxgluserum_mapper)




#A1Cresult
A1Cresult_enc = dataset['A1Cresult'].unique()
'''
labelencoder.fit(['None','Norm','>7','>8'])
dataset['A1Cresult'] = labelencoder.transform(dataset['A1Cresult'])
'''
a1c_mapper = {'None':0,'Norm':1,'>7':2,'>8':3}
dataset['A1Cresult'] = dataset['A1Cresult'].replace(a1c_mapper)




#### Onehot Encoding
onehotencodinglist = ['race','gender','medical_specialty','change','diabetesMed',
                      'diag_1','diag_2','diag_3', 'admission_source_id', 'discharge_disposition_id'
                      ]
def get_dummies_prefix(feature):
    enc = pd.get_dummies(dataset[feature])
    enc.columns = dataset[feature].unique()
    enc_prefix = enc.add_prefix(feature+'_')
    return enc_prefix


# encoded dataframe for each feature    
race_enc = get_dummies_prefix('race')
# randomly drop one column of the dataframe generated by onehot encoding
race_enc = race_enc.drop(race_enc.sample(1,axis = 1).columns, axis = 1)

gender_enc = get_dummies_prefix('gender')
gender_enc = gender_enc.drop(gender_enc.sample(1,axis = 1).columns, axis = 1)

med_specialty_enc = get_dummies_prefix('medical_specialty')
med_specialty_enc = med_specialty_enc.drop(med_specialty_enc.sample(1,axis = 1).columns, axis = 1)

change_enc = get_dummies_prefix('change')
change_enc = change_enc.drop(change_enc.sample(1,axis = 1).columns, axis = 1)

diabetesMed_enc = get_dummies_prefix('diabetesMed')
diabetesMed_enc = diabetesMed_enc.drop(diabetesMed_enc.sample(1,axis = 1).columns, axis = 1)

admission_enc = get_dummies_prefix('admission_source_id')
admission_enc = admission_enc.drop(admission_enc.sample(1,axis = 1).columns, axis = 1)

discharge_enc = get_dummies_prefix('discharge_disposition_id')
discharge_enc = discharge_enc.drop(discharge_enc.sample(1,axis = 1).columns, axis = 1)


##Diag1,2,3 merging categories and encoding
#Diag1 
data_diag = dataset[['diag_1','diag_2','diag_3']]
#converting Vdd and Edd to 9999,floating number
data_diag['diag_1'] = data_diag['diag_1'].str.replace(r'[A-Z]\d+', '9999')
data_diag['diag_2'] = data_diag['diag_2'].str.replace(r'[A-Z]\d+', '9999')
data_diag['diag_3'] = data_diag['diag_3'].str.replace(r'[A-Z]\d+', '9999')


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



# merge all dataframes
new_dataset = pd.concat([dataset,race_enc,gender_enc,med_specialty_enc,change_enc,admission_enc,
                         discharge_enc,diabetesMed_enc, data_diag_], axis = 1)
# remove original feature columns
new_dataset.drop(labels =onehotencodinglist,axis = 1, inplace = True)
new_dataset.to_csv('new_dataset_3.csv')


#############################
dataset.drop(labels = ['diag_1','diag_2','diag_3'], axis = 1, inplace = True)
dataset1130 = pd.concat([dataset, df], axis = 1)
dataset1130.to_csv('dataset_1130.csv')



###########feature engineering################

data = pd.read_csv("new_dataset_3.csv", index_col= 'Unnamed: 0')
index = data['patient_nbr'].index
tf = data['patient_nbr'].value_counts()
unique_ids = data['patient_nbr'].unique()
data = data.sort_values(by = 'encounter_id')
#data.groupby('patient_nbr')
list_of_returning = tf[(tf > 1)].index
#Create a dummy column 
returned = (data[data['patient_nbr'].duplicated(keep = 'first')]).index
data['Returning'] = data.loc[returned].apply(lambda x: 1)
data['Returning'][returned] = 1

data['Returning'] = data['Returning'].fillna(0)
data.sort_index()['Returning'].to_csv('Data/Returned_Feature.csv')
data['Returning'].sort_index().to_csv('Data/Returned_Feature.csv')


########### drop more medications
rawdata = new_dataset
new_feature = (pd.read_csv('Data/Returned_Feature.csv', header = None)).drop(0, axis = 1)

rawdata['admission_type_id'] = rawdata['admission_type_id'].apply(lambda x: 1 if x == 1 else (0 if x == 3 else -1))

d_admission = pd.get_dummies(rawdata['admission_type_id'])

corr = rawdata.corr()
sns.heatmap(corr)
rawdata = rawdata.drop(['encounter_id'], axis = 1)
rawdata = rawdata.drop(['patient_nbr'], axis = 1)
rawdata = rawdata.drop(['admission_type_id'], axis = 1)
rawdata = rawdata.drop(['acetohexamide'], axis = 1) #Majority with the exception of one data point uses this
rawdata = rawdata.drop(['glimepiride-pioglitazone'], axis = 1) #Same as above
rawdata = rawdata.drop(['metformin-rosiglitazone'], axis = 1) #Same as above
rawdata = rawdata.drop(['metformin-pioglitazone'], axis = 1) #Same as above

#d_admission
rawdata['Admission_Emergency'] = d_admission[1]
rawdata['Admission_Elective'] = d_admission[0]
order = ['age', 'time_in_hospital', 'num_lab_procedures', 'num_procedures','max_glu_serum',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses',  'A1Cresult',
       'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
       'glimepiride', 'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'race_AfricanAmerican',
       'race_Other', 'race_Hispanic', 'race_Asian', 'gender_Female',
       'gender_Unknown/Invalid', 'medical_specialty_InternalMedicine',
       'medical_specialty_Cardiology', 'medical_specialty_Surgery',
       'medical_specialty_Others', 'change_Ch',
       'admission_source_id_Admitted from emergency room',
       'admission_source_id_Others',
       'discharge_disposition_id_Discharged to home', 'diabetesMed_Yes',
       'diag_1_Circulatory', 'diag_1_Diabetes', 'diag_1_Digestive',
       'diag_1_Genitourinary', 'diag_1_Injury', 'diag_1_Musculoskeletal',
       'diag_1_Neoplasms', 'diag_1_Other', 'diag_1_Respiratory',
       'diag_2_Circulatory', 'diag_2_Diabetes', 'diag_2_Digestive',
       'diag_2_Genitourinary', 'diag_2_Injury', 'diag_2_Musculoskeletal',
       'diag_2_Neoplasms', 'diag_2_Other', 'diag_2_Respiratory',
       'diag_3_Circulatory', 'diag_3_Diabetes', 'diag_3_Digestive',
       'diag_3_Genitourinary', 'diag_3_Injury', 'diag_3_Musculoskeletal',
       'diag_3_Neoplasms', 'diag_3_Other', 'diag_3_Respiratory', 'Admission_Emergency',
       'Admission_Elective', 'readmitted']

min_max_scaler = preprocessing.MinMaxScaler()
rawdata['readmitted'].sum()
rawdata = rawdata[order]

rawdata['readmitted'].sum()


d_age = pd.get_dummies(rawdata['age'])
rawdata = rawdata.drop('age', axis = 1)
rawdata['readmitted'].sum()
rawdata['Age_Group_2'] = d_age.iloc[:,1]
rawdata['Age_Group_3'] = d_age.iloc[:,2]
rawdata['readmitted'].sum()

order_1 = ['time_in_hospital', 'num_lab_procedures', 'num_procedures', 'max_glu_serum',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses', 'A1Cresult', 'metformin',
       'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
       'glipizide', 'glyburide', 'tolbutamide',
       'pioglitazone', 'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone',
       'tolazamide', 'examide', 'citoglipton', 'insulin',
       'glyburide-metformin', 'glipizide-metformin',
       'race_AfricanAmerican', 'race_Other',
       'race_Hispanic', 'race_Asian', 'gender_Female',
       'gender_Unknown/Invalid', 'medical_specialty_InternalMedicine',
       'medical_specialty_Cardiology', 'medical_specialty_Surgery',
       'medical_specialty_Others', 'change_Ch',
       'admission_source_id_Admitted from emergency room',
       'admission_source_id_Others',
       'discharge_disposition_id_Discharged to home', 'diabetesMed_Yes',
       'diag_1_Circulatory', 'diag_1_Diabetes', 'diag_1_Digestive',
       'diag_1_Genitourinary', 'diag_1_Injury', 'diag_1_Musculoskeletal',
       'diag_1_Neoplasms', 'diag_1_Other', 'diag_1_Respiratory',
       'diag_2_Circulatory', 'diag_2_Diabetes', 'diag_2_Digestive',
       'diag_2_Genitourinary', 'diag_2_Injury', 'diag_2_Musculoskeletal',
       'diag_2_Neoplasms', 'diag_2_Other', 'diag_2_Respiratory',
       'diag_3_Circulatory', 'diag_3_Diabetes', 'diag_3_Digestive',
       'diag_3_Genitourinary', 'diag_3_Injury', 'diag_3_Musculoskeletal',
       'diag_3_Neoplasms', 'diag_3_Other', 'diag_3_Respiratory',
       'Admission_Emergency', 'Admission_Elective',
       'Age_Group_2', 'Age_Group_3', 'readmitted']

rawdata = rawdata[order_1]

corr_post = rawdata.corr()
sns.heatmap(corr_post)


rawdata['A1C_result_norm'] = (pd.get_dummies(rawdata['A1Cresult'])).iloc[:,1]
rawdata['A1C_result_>7'] = (pd.get_dummies(rawdata['A1Cresult'])).iloc[:,2]
rawdata['A1C_result_>8'] = (pd.get_dummies(rawdata['A1Cresult'])).iloc[:,3]
rawdata = rawdata.drop('A1Cresult', axis = 1)

rawdata['max_glu_serum_Norm'] = (pd.get_dummies(rawdata['max_glu_serum'])).iloc[:,1]
rawdata['max_glu_serum_>7'] = (pd.get_dummies(rawdata['max_glu_serum'])).iloc[:,2]
rawdata['max_glu_serum_>8'] = (pd.get_dummies(rawdata['max_glu_serum'])).iloc[:,3]
rawdata = rawdata.drop('max_glu_serum', axis = 1)

list_of_things = ['metformin', 'repaglinide',
       'nateglinide',  'glimepiride', 'glipizide',
       'glyburide',  'pioglitazone', 'rosiglitazone', 
       'insulin',  
       ]

things_to_drop = ['chlorpropamide','tolbutamide','acarbose','miglitol','troglitazone', 'tolazamide'
                                  , 'examide', 'citoglipton', 'glyburide-metformin','glipizide-metformin'
                                  ]


for l in list_of_things:
    print(l)
    rawdata[l+'_No'] = (pd.get_dummies(rawdata[l])).iloc[:,1]
    rawdata[l+'_Down'] = (pd.get_dummies(rawdata[l])).iloc[:,2]
    rawdata[l+'_Steady'] = (pd.get_dummies(rawdata[l])).iloc[:,3]
    rawdata = rawdata.drop(labels = l, axis = 1)



for d in things_to_drop:
    rawdata = rawdata.drop(labels = d, axis = 1)
    
corr_post = rawdata.corr()
sns.heatmap(corr_post)


norm_data = min_max_scaler.fit_transform(rawdata)

norm_data = pd.DataFrame(norm_data)
norm_data.columns = order_post


norm_data = norm_data.drop('gender_Unknown/Invalid' ,axis = 1)

col = norm_data.columns


norm_data['returning'] = new_feature
order_norm = ['time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses', 'race_AfricanAmerican',
       'race_Other', 'race_Hispanic', 'race_Asian', 'gender_Female',
       'medical_specialty_InternalMedicine', 'medical_specialty_Cardiology',
       'medical_specialty_Surgery', 'medical_specialty_Others', 'change_Ch',
       'admission_source_id_Admitted from emergency room',
       'admission_source_id_Others',
       'discharge_disposition_id_Discharged to home', 'diabetesMed_Yes',
       'diag_1_Circulatory', 'diag_1_Diabetes', 'diag_1_Digestive',
       'diag_1_Genitourinary', 'diag_1_Injury', 'diag_1_Musculoskeletal',
       'diag_1_Neoplasms', 'diag_1_Other', 'diag_1_Respiratory',
       'diag_2_Circulatory', 'diag_2_Diabetes', 'diag_2_Digestive',
       'diag_2_Genitourinary', 'diag_2_Injury', 'diag_2_Musculoskeletal',
       'diag_2_Neoplasms', 'diag_2_Other', 'diag_2_Respiratory',
       'diag_3_Circulatory', 'diag_3_Diabetes', 'diag_3_Digestive',
       'diag_3_Genitourinary', 'diag_3_Injury', 'diag_3_Musculoskeletal',
       'diag_3_Neoplasms', 'diag_3_Other', 'diag_3_Respiratory',
       'Admission_Emergency', 'Admission_Elective', 'Age_Group_2',
       'Age_Group_3', 'A1C_result_norm', 'A1C_result_>7', 'A1C_result_>8',
       'max_glu_serum_Norm', 'max_glu_serum_>7', 'max_glu_serum_>8',
       'metformin_No', 'metformin_Down', 'metformin_Steady', 'repaglinide_No',
       'repaglinide_Down', 'repaglinide_Steady', 'nateglinide_No',
       'nateglinide_Down', 'nateglinide_Steady', 'glimepiride_No',
       'glimepiride_Down', 'glimepiride_Steady', 'glipizide_No',
       'glipizide_Down', 'glipizide_Steady', 'glyburide_No', 'glyburide_Down',
       'glyburide_Steady', 'pioglitazone_No', 'pioglitazone_Down',
       'pioglitazone_Steady', 'rosiglitazone_No', 'rosiglitazone_Down',
       'rosiglitazone_Steady', 'insulin_No', 'insulin_Down', 'insulin_Steady','returning',
       'readmitted']

norm_data = norm_data[order_norm]

norm_data.to_csv('Data/Cleaned.csv')


y = norm_data['readmitted']
x = norm_data.iloc[:,:87]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)

X_train['y'] = y_train
X_test['y'] = y_test
X_test.to_csv('Data/New_test_set.csv')
X_train.to_csv('Data/New_train_set.csv')
























