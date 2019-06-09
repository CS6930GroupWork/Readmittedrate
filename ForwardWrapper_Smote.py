#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd

diabetic = pd.read_csv("New_train_set.csv")
diabetic = diabetic.drop(diabetic.columns[0],axis = 1)


# In[3]:


from sklearn.model_selection import KFold
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.metrics import f1_score

def wrapper(clf,X_raw,y,k):
    X = X_raw.copy()
    Wrap_accuracy_old = 0
    Wrap_accuracy_new = 0.00001
    n = 0
    #generate an empty dataframe to save features
    Wrap_filt = X.iloc[:,[0,1]]
    Wrap_filt = Wrap_filt.drop(Wrap_filt.columns[[0,1]],axis=1)
    
    kf = KFold(n_splits=k, random_state=42, shuffle=False)
    sm = SMOTE(random_state = 42, ratio = 1)
    
    while(Wrap_accuracy_old <  Wrap_accuracy_new):
        Wrap_accuracy_old = Wrap_accuracy_new
        avg_accuracies_list = []

        for i in range(X.shape[1]):
            accur_list = []
            for train_index, test_index in kf.split(X):
                
                
                X_res, y_res = sm.fit_sample(Wrap_filt.join(X.iloc[:,i]).iloc[train_index,:], y.iloc[train_index])
                clf.fit(X_res, y_res)
                y_pred = clf.predict(Wrap_filt.join(X.iloc[:,i]).iloc[test_index,:])
                
                score = f1_score(y.iloc[test_index],y_pred,average = "weighted")
                accur_list.append(score)
            avg_accuracies_list.append(np.mean(accur_list))
            
        # add the highest accuracy feature to the selected dataframe
        Wrap_filt = Wrap_filt.join(X.iloc[:,np.argsort(avg_accuracies_list)[-1]])
        # drop the selected feature from the unselected dataframe
        X = X.drop(X.columns[np.argsort(avg_accuracies_list)[-1]],axis=1)
        # renew the accuracy
        Wrap_accuracy_new = np.sort(avg_accuracies_list)[-1]                 
        print(n,"selection is done")
        n = n + 1
    return Wrap_accuracy_old,n-1,Wrap_filt.drop(Wrap_filt.columns[-1],axis=1)


# In[9]:


from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

clf1 = GaussianNB()
clf2 = Perceptron(random_state=42, n_jobs=-1)
clf3 = RandomForestClassifier(n_estimators='warn',n_jobs=-1)
clf4 = LinearSVC()
clf5 = KNeighborsClassifier(n_neighbors=5)
clf6 = LogisticRegression()


# In[7]:


logit_acc, logit_num_feature, logit_features = wrapper(clf6,diabetic.iloc[:,:-1],diabetic.iloc[:,-1],5)


# In[18]:





# In[23]:


logit_features.to_csv("LogitFeatures.csv",index=False)
logit_acc


# In[11]:


NB_acc, NB_num_feature, NB_features = wrapper(clf1,diabetic.iloc[:,:-1],diabetic.iloc[:,-1],5)


# In[24]:


NB_features.to_csv("NBFeatures.csv",index=False)
NB_acc


# In[12]:


Per_acc, Per_num_feature, Per_features = wrapper(clf2,diabetic.iloc[:,:-1],diabetic.iloc[:,-1],5)


# In[25]:


Per_features.to_csv("PerFeatures.csv",index=False)
Per_acc


# In[13]:


RF_acc, RF_num_feature, RF_features = wrapper(clf3,diabetic.iloc[:,:-1],diabetic.iloc[:,-1],5)


# In[26]:


RF_features.to_csv("RFFeatures.csv",index=False)
RF_acc


# In[14]:


SVM_acc, SVM_num_feature, SVM_features = wrapper(clf4,diabetic.iloc[:,:-1],diabetic.iloc[:,-1],5)


# In[27]:


SVM_features.to_csv("SVMFeatures.csv",index=False)
SVM_acc


# In[15]:


knn_acc, knn_num_feature, knn_features = wrapper(clf5,diabetic.iloc[:,:-1],diabetic.iloc[:,-1],5)


# In[28]:


knn_features.to_csv("knnFeatures.csv",index=False)
knn_acc


# In[31]:


modelname = ['NB',"Perceptron","RF","SVM","KNN","Logit"]
summary_acc = pd.DataFrame({"accuracy":[NB_acc,Per_acc,RF_acc,SVM_acc,knn_acc,logit_acc]},index=modelname)
summary_acc.to_csv("Accuracy.csv")

