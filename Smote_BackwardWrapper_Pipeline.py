#!/usr/bin/env python
# coding: utf-8

# In[1]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import csv


# In[2]:


import pandas as pd
import numpy as np

diabetic = pd.read_csv("New_train_set.csv")
diabetic_test = pd.read_csv("New_test_set.csv")
diabetic = diabetic.drop(diabetic.columns[0],axis = 1)
diabetic_test = diabetic_test.drop(diabetic_test.columns[0],axis = 1)


# In[3]:


def sm_col_clf_piper(X_train, y_train, X_test, X_label, parameters, clf, scoring= 'f1'):
    
    # parameters: dict of parameter need to tune
    # clf: classifier
    # n_features: number of features want to find, default is half of all features
    # scoring: type of score using to tune, default is f1 score
    
    # SMOTE training set to deal with imbalance
    sm = SMOTE()
    X_train, X_label = sm.fit_sample(X_train, X_label)
    
    pipe = make_pipeline(
                    (SFS(clf,"best",forward=False,scoring=scoring,cv=5)),
                    (clf)
                    )
    # tune model with different parameters
    grid = GridSearchCV(estimator = pipe, param_grid = parameters, cv = 5, n_jobs = -1, verbose = 50, scoring = scoring)
    grid.fit(X_train, y_train)
    # get the selected feature index 
    best_pipe = grid.best_estimator_
    feature_idx = (best_pipe.named_steps['sequentialfeatureselector'].transform(np.arange(len(X_train.columns)).reshape(1, -1)))[0]
    # use best parameter to predict test label
    pred = grid.predict(X_test)
    
    # calculate different score based on prediction
    conf = confusion_matrix(X_label,pred)
    test_score = {
    "accuracy":accuracy_score(X_label,pred),
    "precision":precision_score(X_label,pred,"binary"),
    "recall":recall_score(X_label,pred,"binary"),
    "f1_score":f1_score(X_label,pred,"binary"),
    "roc_auc":roc_auc_score(X_label,pred)
    }
    return grid.cv_results_['mean_test_score'], grid.best_params_, conf , test_score, feature_idx


# In[4]:


def write_csv(name, parameters):
    with open(name + '_param.csv', 'w') as f:
        for key in parameters.keys():
            f.write("%s,%s\n"%(key, parameters[key]))


# In[5]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

clf1 = GaussianNB()
clf2 = Perceptron(tol=1e-3, random_state=42)
clf3 = RandomForestClassifier(n_estimators='warn', criterion='gini', max_depth=None, 
                          min_samples_split=2, min_samples_leaf=1, 
                          min_weight_fraction_leaf=0.0, max_features='auto', 
                          max_leaf_nodes=None, min_impurity_decrease=0.0, 
                          min_impurity_split=None, bootstrap=True, oob_score=False, 
                          n_jobs=None, random_state=None, verbose=0, warm_start=False, 
                          class_weight=None)
clf4_5 = LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=3)
clf4 = SVC(verbose=3)
clf5 = KNeighborsClassifier(n_neighbors=5)
clf6 = LogisticRegression(solver='liblinear')


# In[6]:


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


# In[ ]:


logit_score, logit_param, logit_conf, logit_test_score, logit_feature_idx = sm_col_clf_piper(diabetic.iloc[:,:-1],diabetic['y'],diabetic_test.iloc[:,:-1],diabetic_test['y'],parameters_log,clf6)
print("Train score:",logit_score)
print("Tuned parameter:",logit_param)
print("Confusion Matrix:",logit_conf)
print("Test score:",logit_test_score)


# In[ ]:


write_csv('logit_para', logit_param)
write_csv('logit_test_score', logit_test_score)
(diabetic.iloc[:,logit_feature_idx]).to_csv("logit_feature.csv")


# In[ ]:


per_score, per_param, per_conf, per_test_score, per_feature_idx = sm_col_clf_piper(diabetic.iloc[:,:-1],diabetic['y'],diabetic_test.iloc[:,:-1],diabetic_test['y'],parameters_per,clf2)
print("Train score:",per_score)
print("Tuned parameter:",per_param)
print("Confusion Matrix:",per_conf)
print("Test score:",per_test_score)


# In[ ]:


write_csv('per_para', per_param)
write_csv('per_test_score', per_test_score)
(diabetic.iloc[:,per_feature_idx]).to_csv("per_feature.csv")


# In[ ]:


lin_svm_score, lin_svm_param, lin_svm_conf, lin_svm_test_score, lin_svm_feature_idx = sm_col_clf_piper(diabetic.iloc[:,:-1],diabetic['y'],diabetic_test.iloc[:,:-1],diabetic_test['y'],parameters_lin_svm,clf4_5)
print("Train score:",lin_svm_score)
print("Tuned parameter:",lin_svm_param)
print("Confusion Matrix:",lin_svm_conf)
print("Test score:",lin_svm_test_score)


# In[ ]:


write_csv('lin_svm_para', lin_svm_param)
write_csv('lin_svm_test_score', lin_svm_test_score)
(diabetic.iloc[:,lin_svm_feature_idx]).to_csv("lin_svm_feature.csv")


# In[ ]:


RF_score, RF_param, RF_conf, RF_test_score, RF_feature_idx = sm_col_clf_piper(diabetic.iloc[:,:-1],diabetic['y'],diabetic_test.iloc[:,:-1],diabetic_test['y'],c45_param,clf3)
print("Train score:",RF_score)
print("Tuned parameter:",RF_param)
print("Confusion Matrix:",RF_conf)
print("Test score:",RF_test_score)


# In[ ]:


write_csv('RF_para', RF_param)
write_csv('RF_test_score', RF_test_score)
(diabetic.iloc[:,RF_feature_idx]).to_csv("RF_feature.csv")


# In[ ]:


knn_score, knn_param, knn_conf, knn_test_score, knn_feature_idx = sm_col_clf_piper(diabetic.iloc[:,:-1],diabetic['y'],diabetic_test.iloc[:,:-1],diabetic_test['y'],parameters_knn,clf5)
print("Train score:",knn_score)
print("Tuned parameter:",knn_param)
print("Confusion Matrix:",knn_conf)
print("Test score:",knn_test_score)


# In[ ]:


write_csv('knn_para', knn_param)
write_csv('knn_test_score', knn_test_score)
(diabetic.iloc[:,knn_feature_idx]).to_csv("knn_feature.csv")


# In[7]:


pipe_NB = make_pipeline(
                (SFS(clf1,"best",forward=False,scoring='f1',cv=5)),
                (clf1)
                )

pipe_NB.fit(diabetic.iloc[:,:-1], diabetic['y'])
# get the selected feature index 
NB_feature_idx = (pipe_NB.named_steps['sequentialfeatureselector'].transform(np.arange(len((diabetic.iloc[:,:-1]).columns)).reshape(1, -1)))[0]
# use best parameter to predict test label
pred_NB = pipe_NB.predict(diabetic_test.iloc[:,:-1])

# calculate different score based on prediction
NB_conf = confusion_matrix(diabetic_test['y'],pred_NB)
NB_test_score = {
"accuracy":accuracy_score(diabetic_test['y'],pred_NB),
"precision":precision_score(diabetic_test['y'],pred_NB,"binary"),
"recall":recall_score(diabetic_test['y'],pred_NB,"binary"),
"f1_score":f1_score(diabetic_test['y'],pred_NB,"binary"),
"roc_auc":roc_auc_score(diabetic_test['y'],pred_NB)
}


# In[8]:


write_csv('NB_test_score', NB_test_score)
(diabetic.iloc[:,NB_feature_idx]).to_csv("NB_feature.csv")

