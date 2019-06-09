#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import enum


# In[2]:


#Feature Scoring Method:
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif


# In[3]:


#Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier


# In[4]:


from sklearn.model_selection import KFold


# In[5]:


from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix


# In[6]:


#Data Imbalence Resolver
from imblearn.over_sampling import SMOTE


# In[7]:


from sklearn import preprocessing


# In[8]:


#Math Formulas
import math


# In[9]:


from scipy.stats import pearsonr


# In[10]:


import matplotlib.pyplot as plt


# In[11]:


#Put this in global
sm = SMOTE(random_state = 42)
kf = KFold(n_splits=5, random_state=42, shuffle=False)


# In[12]:


train = pd.read_csv('Data/New_train_set.csv', index_col='Unnamed: 0')
test = pd.read_csv('Data/New_test_set.csv', index_col='Unnamed: 0')


# In[13]:


X_train = train.iloc[:,:87]
y_train = train['y']


# In[24]:


X_train


# In[15]:


#Models to Run
clf1 = LogisticRegression()
clf2 = Perceptron()
clf3 = GaussianNB()
clf4 = RandomForestClassifier()
clf5 = LinearSVC()
clf6 = KNeighborsClassifier()


# In[16]:


#Returns Feature Importance:
#Takes input k to score for "important" features
def get_importance_score(k, score_func, X_train, Y_train):
    k_best = SelectKBest(score_func = score_func, k = k)
    fit = k_best.fit(X_train, Y_train)
    scores = pd.Series(index = X_train.columns, data = fit.scores_)
    return scores

def get_avg_feat_importance(clf, X_train, y_train):
    accur_list = []
    feat_import = []
    sum_of_acc = 0
    for train_index, test_index in kf.split(X_train):
        #Split training value between 1 test and 4 train folds
        k_X_train, k_X_test = X_train.iloc[train_index], X_train.iloc[test_index]
        k_y_train, k_y_test = y_train.iloc[train_index], y_train.iloc[test_index]
        #Smote training sets
        sm_X_train, sm_y_train = sm.fit_sample(k_X_train, k_y_train)
        clf.fit(X = k_X_train, y= k_y_train)
        k_pred_y = clf.predict(k_X_test)
        accur_list.append(f1_score(y_true = k_y_test, y_pred = k_pred_y, average = 'macro'))
        feat_import.append(clf.feature_importances_)
    avg_accur = sum(accur_list)/len(accur_list)
    avg_feat_im = sum(feat_import)/len(accur_list)
    for i in range(len(accur_list)):
        sum_of_acc = sum_of_acc +(accur_list[0] - avg_accur)**2
    std = math.sqrt((sum_of_acc)/len(accur_list))
    print("The average accuracy is: ",avg_accur, "and the std is: ", std)
    return (avg_feat_im)


# In[17]:


sm_X_train, sm_y_train = sm.fit_sample(X_train, y_train)
correlation_list = []
sm_X_train = pd.DataFrame(sm_X_train)
for i in range(sm_X_train.shape[1]):
    correlation_list.append(pearsonr(sm_X_train.iloc[:,i],sm_y_train)[0])


# In[18]:


k = sm_X_train.shape[1]
chi_scoring = get_importance_score(k, chi2, sm_X_train, sm_y_train)
corr_scoring = pd.Series(correlation_list)
#corr_scoring = get_importance_score(k, f_classif, X_train, y_train)
mi_scoring =  get_importance_score(k, mutual_info_classif, sm_X_train, sm_y_train)
#rand_for_scores = get_avg_feat_importance(clf4, X_train, y_train)


# In[19]:


sm_X_train.columns


# In[20]:


scores = pd.DataFrame(data = dict( corr = corr_scoring.values, chi2 = chi_scoring.values 
                                                , MI = mi_scoring.values))#rand_for = rand_for_scores,
column_order = ['Feature_Name', 'corr', 'chi2', 'MI', 'tot_mean'] #'rand_for_imp'
#print(scores)
scores['corr'] = scores['corr'].apply(abs)
#print(scores)
min_max_scaler = preprocessing.MinMaxScaler()
scores = pd.DataFrame(min_max_scaler.fit_transform(scores.iloc[:,:]))
scores['tot_mean'] = scores.mean(axis = 1)
scores['Feature_Name'] = X_train.columns
scores.columns = ['corr', 'chi2', 'MI', 'tot_mean', 'Feature_Name'] # 'rand_for_imp'
scores = scores[column_order]
scores['corr_Rank'] = scores['corr'].rank(ascending = False)
scores['chi2_Rank'] = scores['chi2'].rank(ascending = False)
scores['MI_Rank'] = scores['MI'].rank(ascending = False)
#scores['rand_for_imp_Rank'] = scores['rand_for_imp'].rank(ascending = False)


# In[21]:


scores.to_csv('Data/Scores.csv')


# In[22]:


score_values = scores.columns[1:4]
rank_values = scores.columns[5:]


# In[23]:


for i in range(3):
    plt.plot(scores[score_values[i]], scores[rank_values[i]], 'o')
plt.legend(labels = ('Correlation', 'Chi Squared', 'Mutual Information',))
plt.show()
plt.savefig('Data/Base_Scoring_CFA.png')


# In[ ]:





# In[ ]:





# In[ ]:




