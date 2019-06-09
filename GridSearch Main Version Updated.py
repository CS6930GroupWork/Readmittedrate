#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


from sklearn.model_selection import GridSearchCV


# In[3]:


from mlxtend.feature_selection import ColumnSelector


# In[4]:


#Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[5]:


#SMOTE things
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE


# In[6]:


import csv


# In[7]:


clf1 = LogisticRegression(solver = 'liblinear')
clf2 = Perceptron(tol=1e-3, random_state=42)
clf3 = GaussianNB()

clf4_5 = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=3)

clf4 = SVC(verbose=3)
clf5 = RandomForestClassifier(n_estimators='warn', criterion='gini', max_depth=None, 
                          min_samples_split=2, min_samples_leaf=1, 
                          min_weight_fraction_leaf=0.0, max_features='auto', 
                          max_leaf_nodes=None, min_impurity_decrease=0.0, 
                          min_impurity_split=None, bootstrap=True, oob_score=False, 
                          n_jobs=1, random_state=None, verbose=0, warm_start=False, 
                          class_weight=None)
clf6 = KNeighborsClassifier(n_neighbors=5)


# In[8]:


#Read the dataset
data = pd.read_csv("Data/New_train_set.csv",index_col= 'Unnamed: 0')


# In[9]:


#General Model reading all and spliting x and y from each other
X_train = data.iloc[:,:87]
y_train = data['y']


# In[10]:


#Filter Forward Bin is finished
"""
#This is binary logit wrapper 
#This reads in the columns from the wrapper method
logit_columns = pd.read_csv("Forward/Binary Results/LogitFeatures.csv").columns
percept_columns = pd.read_csv("Forward/Binary Results/PerFeatures.csv").columns
#Naive Bayes is skipped there isnt any tunning available
SVM_columns = pd.read_csv("Forward/Binary Results/SVMFeatures.csv").columns
C45_columns = pd.read_csv("Forward/Binary Results/RFFeatures.csv").columns
knn_columns = pd.read_csv("Forward/Binary Results/knnFeatures.csv").columns
"""


# In[11]:


w_path = 'Wrapper Features/'


# In[12]:


"""
#Wrapper Aproach
Logit_Features = pd.read_csv(w_path + "LogitFeatures.csv").columns
Percept_Features = pd.read_csv(w_path + "PerFeatures.csv").columns
NB_Features = pd.read_csv(w_path + "NBFeatures.csv").columns
SVM_Features = pd.read_csv(w_path + "SVMFeatures.csv").columns
C45_Features = pd.read_csv(w_path + "RFFeatures.csv").columns
KNN_Features = pd.read_csv(w_path + "knnFeatures.csv").columns
"""


# In[13]:


f_path = '/home/jho9/Documents/James Code/Filter_Scoring'


# In[14]:


df = pd.read_csv(f_path + '/Scores.csv', index_col= 'Unnamed: 0')


# In[15]:


df['Rank_mean'] = df[['corr_Rank', 'chi2_Rank', 'MI_Rank']].mean(axis = 1)


# In[16]:


#Filter Feature Scoring Methods:
#Corr_Features = ((pd.read_csv(f_path + '/Scores.csv', index_col= 'Unnamed: 0')).sort_values(by = 'corr_Rank')).iloc[:10,:]['Feature_Name'].tolist()
#Chi_Features = ((pd.read_csv(f_path + '/Scores.csv', index_col= 'Unnamed: 0'))).sort_values(by = 'chi2_Rank').iloc[:10,:]['Feature_Name'].tolist()
MI_Features = ((pd.read_csv(f_path + '/Scores.csv', index_col= 'Unnamed: 0'))).sort_values(by = 'MI_Rank').iloc[:10,:]['Feature_Name'].tolist()
AVG_Features = ((pd.read_csv(f_path + '/Scores.csv', index_col= 'Unnamed: 0'))).sort_values(by = 'tot_mean', ascending = False).iloc[:10,:]['Feature_Name'].tolist()
AVG_rank_Features = df.sort_values(by = 'Rank_mean').iloc[:10,:]['Feature_Name'].tolist()


# In[17]:


def write_csv(name, parameters):
    with open(name + '_param.csv', 'w') as f:
        for key in parameters.keys():
            f.write("%s,%s\n"%(key, parameters[key]))


# In[18]:


def sm_col_clf_piper(X_train, y_train, parameters, column, clf, scoring= 'f1', n_jobs = 6):
    pipe = make_pipeline(
                    (ColumnSelector(cols = column)),
                    (SMOTE()),
                    (clf)
    )
    print(pipe.get_params().keys())
    grid = GridSearchCV(estimator = pipe, param_grid = parameters, cv = 5, n_jobs = n_jobs, verbose = 50, scoring = scoring)
    grid.fit(X_train, y_train)
    return grid.cv_results_['mean_test_score'], grid.best_params_


# In[19]:


#Logit Tuning Params
C = [ 0.01, 0.1, 1, 10, 100] 
parameters_log = {'logisticregression__C':C}
#Percept Tuning Params
alpha = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000]
parameters_per = {'perceptron__alpha':alpha}
#Lin SVM Tunning Params
parameters_lin_svm = {'linearsvc__C': C}
#SVM tuning Params
degree = [1,2]
gamma = [.001, .01,]
kernel = ["rbf","poly"]
parameters_svm = {'svc__C':C, 'svc__degree': degree, 'svc__gamma':gamma,'svc__kernel':kernel }
#Random Forest Tunning Params
c45_param = {'randomforestclassifier__n_estimators': [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]}
#KNN Tuning Params
k = [1, 3, 5, 11, 21, 41, 61, 81]
parameters_knn = {'kneighborsclassifier__n_neighbors':k}


# In[20]:


scores_logit, param_logit = sm_col_clf_piper(X_train, y_train, parameters=parameters_log, column = Logit_Features, clf= clf1, scoring='roc_auc')


# In[ ]:


write_csv('logit_wrap_param', param_logit)


# In[ ]:


scores_per, param_per = sm_col_clf_piper(X_train, y_train, parameters=parameters_per, column = Percept_Features, clf= clf2, scoring='roc_auc')


# In[ ]:


write_csv('Perc_wrap_param', param_per)


# In[ ]:


scores_lin_svm, param_lin_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_lin_svm, column = SVM_Features, clf= clf4_5, scoring='roc_auc')


# In[ ]:


write_csv('lin_svm_wrap', param_lin_svm)


# In[ ]:


scores_C45, param_C45 = sm_col_clf_piper(X_train, y_train, parameters=c45_param, column = C45_Features, clf= clf5, scoring='roc_auc')


# In[ ]:


write_csv('C45 Param', param_C45)


# In[ ]:


scores_knn, param_knn = sm_col_clf_piper(X_train, y_train, parameters=parameters_knn, column = KNN_Features, clf= clf6, scoring='roc_auc')


# In[ ]:


write_csv('KNN Param', param_knn)


# In[ ]:


"""
#Filter Feature Scoring Methods:
Corr_features = ((pd.read_csv(f_path + '/Scores.csv', index_col= 'Unnamed: 0')).sort_values(by = 'corr_Rank')).iloc[:10,:]['Feature_Name'].tolist()
Chi_features = ((pd.read_csv(f_path + '/Scores.csv', index_col= 'Unnamed: 0'))).sort_values(by = 'chi2_Rank').iloc[:10,:]['Feature_Name'].tolist()
MI_features = ((pd.read_csv(f_path + '/Scores.csv', index_col= 'Unnamed: 0'))).sort_values(by = 'MI_Rank').iloc[:10,:]['Feature_Name'].tolist()
AVG_features = ((pd.read_csv(f_path + '/Scores.csv', index_col= 'Unnamed: 0'))).sort_values(by = 'tot_mean', ascending = False).iloc[:10,:]['Feature_Name'].tolist()
AVG_rank_features = df.sort_values(by = 'Rank_mean').iloc[:10,:]['Feature_Name'].tolist()
"""


# In[ ]:


scores_logit, param_logit = sm_col_clf_piper(X_train, y_train, parameters=parameters_log, column = Corr_Features, clf= clf1, scoring='roc_auc')
write_csv('logit_corr_param', param_logit)


# In[ ]:


scores_per, param_per = sm_col_clf_piper(X_train, y_train, parameters=parameters_per, column = Corr_Features, clf= clf2, scoring='roc_auc')
write_csv('Perc_corr_param', param_per)


# In[ ]:


scores_lin_svm, param_lin_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_lin_svm, column = Corr_Features, clf= clf4_5, scoring='roc_auc')
write_csv('lin_svm_corr', param_lin_svm)


# In[ ]:


scores_C45, param_C45 = sm_col_clf_piper(X_train, y_train, parameters=c45_param, column = Corr_Features, clf= clf5, scoring='roc_auc')
write_csv('C45_corr', param_C45)


# In[ ]:


scores_knn, param_knn = sm_col_clf_piper(X_train, y_train, parameters=parameters_knn, column = Corr_Features, clf= clf6, scoring='roc_auc')
write_csv('KNN_corr', param_knn)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


scores_logit, param_logit = sm_col_clf_piper(X_train, y_train, parameters=parameters_log, column = Chi_Features, clf= clf1, scoring='roc_auc')
write_csv('logit_chi_param', param_logit)


# In[ ]:


scores_per, param_per = sm_col_clf_piper(X_train, y_train, parameters=parameters_per, column = Chi_Features, clf= clf2, scoring='roc_auc')
write_csv('Perc_chi_param', param_per)


# In[ ]:


scores_lin_svm, param_lin_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_lin_svm, column = Chi_Features, clf= clf4_5, scoring='roc_auc')
write_csv('lin_svm_chi', param_lin_svm)


# In[ ]:


scores_C45, param_C45 = sm_col_clf_piper(X_train, y_train, parameters=c45_param, column = Chi_Features, clf= clf5, scoring='roc_auc')
write_csv('C45_chi', param_C45)


# In[ ]:


scores_knn, param_knn = sm_col_clf_piper(X_train, y_train, parameters=parameters_knn, column = Chi_Features, clf= clf6, scoring='roc_auc')
write_csv('KNN_chi', param_knn)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


scores_logit, param_logit = sm_col_clf_piper(X_train, y_train, parameters=parameters_log, column = MI_Features, clf= clf1, scoring='roc_auc')
write_csv('logit_mi_param', param_logit)


# In[ ]:


scores_per, param_per = sm_col_clf_piper(X_train, y_train, parameters=parameters_per, column = MI_Features, clf= clf2, scoring='roc_auc')
write_csv('Perc_mi_param', param_per)


# In[ ]:


scores_lin_svm, param_lin_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_lin_svm, column = MI_Features, clf= clf4_5, scoring='roc_auc')
write_csv('lin_svm_mi', param_lin_svm)


# In[18]:


scores_C45, param_C45 = sm_col_clf_piper(X_train, y_train, parameters=c45_param, column = MI_Features, clf= clf5, scoring='roc_auc', n_jobs = 1)
write_csv('C45_mi', param_C45)


# In[21]:


scores_knn, param_knn = sm_col_clf_piper(X_train, y_train, parameters=parameters_knn, column = MI_Features, clf= clf6, scoring='roc_auc')
write_csv('KNN_mi', param_knn)


# In[ ]:





# In[ ]:





# In[22]:


scores_logit, param_logit = sm_col_clf_piper(X_train, y_train, parameters=parameters_log, column = AVG_Features, clf= clf1, scoring='roc_auc')
write_csv('logit_avg_score', param_logit)


# In[ ]:



scores_per, param_per = sm_col_clf_piper(X_train, y_train, parameters=parameters_per, column = AVG_Features, clf= clf2, scoring='roc_auc')
write_csv('Perc_avg_score', param_per)

scores_lin_svm, param_lin_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_lin_svm, column = AVG_Features, clf= clf4_5, scoring='roc_auc')
write_csv('lin_svm_avg_score', param_lin_svm)

scores_C45, param_C45 = sm_col_clf_piper(X_train, y_train, parameters=c45_param, column = AVG_Features, clf= clf5, scoring='roc_auc', n_jobs = 1)
write_csv('C45_avg_score', param_C45)

scores_knn, param_knn = sm_col_clf_piper(X_train, y_train, parameters=parameters_knn, column = AVG_Features, clf= clf6, scoring='roc_auc')
write_csv('KNN_avg_score', param_knn)


# In[ ]:





# In[ ]:





# In[ ]:


scores_logit, param_logit = sm_col_clf_piper(X_train, y_train, parameters=parameters_log, column = AVG_rank_Features, clf= clf1, scoring='roc_auc')
write_csv('logit_avg_rank', param_logit)


# In[ ]:



scores_per, param_per = sm_col_clf_piper(X_train, y_train, parameters=parameters_per, column = AVG_rank_Features, clf= clf2, scoring='roc_auc')
write_csv('Perc_avg_rank', param_per)

scores_lin_svm, param_lin_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_lin_svm, column = AVG_rank_Features, clf= clf4_5, scoring='roc_auc')
write_csv('lin_svm_avg_rank', param_lin_svm)

scores_C45, param_C45 = sm_col_clf_piper(X_train, y_train, parameters=c45_param, column = AVG_Features, clf= clf5, scoring='roc_auc', n_jobs = 1)
write_csv('C45_avg_rank', param_C45)

scores_knn, param_knn = sm_col_clf_piper(X_train, y_train, parameters=parameters_knn, column = AVG_Features, clf= clf6, scoring='roc_auc')
write_csv('KNN_avg_rank', param_knn)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


scores_lin_svm, param_lin_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_lin_svm, column = scores, clf= clf4_5, scoring='roc_auc')


# In[ ]:


write_csv('lin_svm_corr', param_lin_svm)


# In[ ]:


#scores_svm, param_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_svm, column = corr_columns, clf= clf4, scoring='f1')


# In[ ]:


#write_csv('svm_corr',param_svm)


# In[ ]:


scores_c45, param_c45 = sm_col_clf_piper(X_train, y_train, parameters=c45_param, column=s, clf= clf5, scoring='roc_auc')


# In[ ]:


write_csv('C45_corr',param_c45)


# In[ ]:


knn_scores, knn_param = sm_col_clf_piper(X_train, y_train, parameters = parameters_knn, column=corr_columns, clf = clf6)


# In[ ]:


write_csv('knn_corr', knn_param)


# In[ ]:


scores_logit, param_logit = sm_col_clf_piper(X_train, y_train, parameters_log, column = chi_columns, clf = clf1, scoring = 'f1')


# In[ ]:


write_csv('logit_chi', param_logit)


# In[ ]:


scores_percept, param_percept = sm_col_clf_piper(X_train, y_train, parameters=parameters_per, column = chi_columns, clf= clf2, scoring='f1')


# In[ ]:


write_csv('percept_chi', param_percept)


# In[ ]:


scores_lin_svm, param_lin_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_lin_svm, column = chi_columns, clf= clf4_5, scoring='f1')


# In[ ]:


write_csv('lin_svm_chi', param_lin_svm)


# In[ ]:


#scores_svm, param_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_svm, column = corr_columns, clf= clf4, scoring='f1')


# In[ ]:


#write_csv('svm_corr',param_svm)


# In[ ]:


scores_c45, param_c45 = sm_col_clf_piper(X_train, y_train, parameters=c45_param, column=chi_columns, clf= clf5, scoring='f1')


# In[ ]:


write_csv('C45_chi',param_c45)


# In[ ]:


knn_scores, knn_param = sm_col_clf_piper(X_train, y_train, parameters = parameters_knn, column=chi_columns, clf = clf6)


# In[ ]:


write_csv('knn_chi', knn_param)


# In[ ]:


scores_logit, param_logit = sm_col_clf_piper(X_train, y_train, parameters_log, column = MI_columns, clf = clf1, scoring = 'f1')


# In[ ]:


write_csv('logit_mi', param_logit)


# In[ ]:


scores_percept, param_percept = sm_col_clf_piper(X_train, y_train, parameters=parameters_per, column = MI_columns, clf= clf2, scoring='f1')


# In[ ]:


write_csv('percept_mi', param_percept)


# In[ ]:


scores_lin_svm, param_lin_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_lin_svm, column = MI_columns, clf= clf4_5, scoring='f1')


# In[ ]:


write_csv('lin_svm_mi', param_lin_svm)


# In[ ]:


#scores_svm, param_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_svm, column = corr_columns, clf= clf4, scoring='f1')


# In[ ]:


#write_csv('svm_corr',param_svm)


# In[ ]:


scores_c45, param_c45 = sm_col_clf_piper(X_train, y_train, parameters=c45_param, column=MI_columns, clf= clf5, scoring='f1')


# In[ ]:


write_csv('C45_MI',param_c45)


# In[ ]:


knn_scores, knn_param = sm_col_clf_piper(X_train, y_train, parameters = parameters_knn, column=MI_columns, clf = clf6)


# In[ ]:


write_csv('knn_MI', knn_param)


# In[ ]:


scores_logit, param_logit = sm_col_clf_piper(X_train, y_train, parameters_log, column = avg_columns, clf = clf1, scoring = 'f1')


# In[ ]:


write_csv('logit_avg', param_logit)


# In[ ]:


scores_percept, param_percept = sm_col_clf_piper(X_train, y_train, parameters=parameters_per, column = avg_columns, clf= clf2, scoring='f1')


# In[ ]:


write_csv('percept_avg', param_percept)


# In[ ]:


scores_lin_svm, param_lin_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_lin_svm, column = avg_columns, clf= clf4_5, scoring='f1')


# In[ ]:


write_csv('lin_svm_avg', param_lin_svm)


# In[ ]:


#scores_svm, param_svm = sm_col_clf_piper(X_train, y_train, parameters=parameters_svm, column = avg_columns, clf= clf4, scoring='f1')


# In[ ]:


#write_csv('svm_avg',param_svm)


# In[ ]:


scores_c45, param_c45 = sm_col_clf_piper(X_train, y_train, parameters=c45_param, column=avg_columns, clf= clf5, scoring='f1')


# In[ ]:


write_csv('C45_avg',param_c45)


# In[ ]:


knn_scores, knn_param = sm_col_clf_piper(X_train, y_train, parameters = parameters_knn, column=avg_columns, clf = clf6)


# In[ ]:


write_csv('knn_avg', knn_param)


# In[ ]:





# In[ ]:




