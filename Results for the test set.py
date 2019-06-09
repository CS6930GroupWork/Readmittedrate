#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


#Model Imports
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


# In[3]:


from mlxtend.feature_selection import ColumnSelector


# In[4]:


from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE


# In[5]:


from sklearn.ensemble import AdaBoostClassifier


# In[6]:


from mlxtend.classifier import EnsembleVoteClassifier


# In[7]:


import csv


# In[8]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn import model_selection
from sklearn.metrics import confusion_matrix


# In[9]:


"""
#Tunned to AUC
tunning_Wrap_path = "C://Users//James//Documents//Updated Again//James Code//Tunning Parameters//Wrapper Binary Tunning Parameters//"
"""


# In[10]:


"""
logit_c = pd.read_csv(tunning_Wrap_path + "logit_param.csv",header=None)
perceptron_alpha = pd.read_csv(tunning_Wrap_path + "percept_param.csv",header=None)
rand_for = pd.read_csv(tunning_Wrap_path + "C45_param.csv",header = None)
lin_svm = pd.read_csv(tunning_Wrap_path + "lin_svm_param.csv", header = None)
knn_k = pd.read_csv(tunning_Wrap_path + "knn_param.csv", header = None)
"""


# In[13]:


#Tunned to AUC
tunning_avg_path = "C://Users//James//Documents//Updated Again//James Code//Tunning Parameters//AUC Parameters//Top 10 AUC//AVG Rank Parameters//"


# In[14]:


#Binary top 10 chi parameters tuned to f1
#NB has no tunning path
logit_c = pd.read_csv(tunning_avg_path + "logit_avg_rank_param.csv",header=None)
perceptron_alpha = pd.read_csv(tunning_avg_path + "Perc_avg_rank_param.csv",header=None)
rand_for = pd.read_csv(tunning_avg_path + "C45_avg_rank_param.csv",header = None)
lin_svm = pd.read_csv(tunning_avg_path + "lin_svm_avg_rank_param.csv", header = None)
knn_k = pd.read_csv(tunning_avg_path + "knn_avg_rank_param.csv", header = None)


# In[15]:


clf1 = LogisticRegression(solver = 'liblinear', C = float(logit_c[1]))
clf2 = Perceptron(tol=1e-3, random_state=42, alpha = float(perceptron_alpha[1]))
clf3 = GaussianNB()

clf4a = LinearSVC(C=float(lin_svm[1]), class_weight=None, dual=True, fit_intercept=True,
     intercept_scaling=1, loss='squared_hinge', max_iter=1000,
     multi_class='ovr', penalty='l2', random_state=0, tol=1e-05, verbose=0)

#clf4 = SVC(verbose=3)
clf5 = RandomForestClassifier(n_estimators=int(rand_for[1]), criterion='gini', max_depth=None, 
                          min_samples_split=2, min_samples_leaf=1, 
                          min_weight_fraction_leaf=0.0, max_features='auto', 
                          max_leaf_nodes=None, min_impurity_decrease=0.0, 
                          min_impurity_split=None, bootstrap=True, oob_score=False, 
                          n_jobs=None, random_state=None, verbose=1, warm_start=False, 
                          class_weight=None)
clf6 = KNeighborsClassifier(n_neighbors=int(knn_k[1]))


# In[16]:


#Read the dataset
data = pd.read_csv("Data/New_train_set.csv",index_col= 'Unnamed: 0')
#General Model reading all and spliting x and y from each other
X_train = data.iloc[:,:87]
y_train = data['y']


# In[17]:


test = pd.read_csv("Data/New_test_set.csv",index_col= 'Unnamed: 0')
#General Model reading all and spliting x and y from each other
X_test = test.iloc[:,:87]
y_test = test['y']


# In[18]:


#Features 
"""
feat_path = "Wrapper Features/"
"""


# In[19]:


"""
logit_feats = pd.read_csv(feat_path + "LogitFeatures.csv").columns.tolist()
percept_feats = pd.read_csv(feat_path + "PerFeatures.csv").columns.tolist()
nb_feats = pd.read_csv(feat_path + "NBFeatures.csv").columns.tolist()
lin_svm_feats = pd.read_csv(feat_path + "SVMFeatures.csv").columns.tolist()
c45_feats = pd.read_csv(feat_path + "RFFeatures.csv").columns.tolist()
knn_feats = pd.read_csv(feat_path + "knnFeatures.csv").columns.tolist()
"""


# In[28]:


feat_filt_path = "C://Users//James//Documents//Updated Again//James Code//Filter_Scoring//Scores.csv"


# In[55]:


#Filter Features
corr_feats = pd.read_csv(feat_filt_path, index_col='Unnamed: 0').sort_values(by = 'corr_Rank').iloc[:10,:]['Feature_Name'].tolist()
chi_feats = pd.read_csv(feat_filt_path, index_col='Unnamed: 0').sort_values(by = 'chi2_Rank').iloc[:10,:]['Feature_Name'].tolist()
mi_feats = pd.read_csv(feat_filt_path, index_col='Unnamed: 0').sort_values(by = 'MI_Rank').iloc[:10,:]['Feature_Name'].tolist()
avg_feats = pd.read_csv(feat_filt_path, index_col='Unnamed: 0').loc[((pd.read_csv(feat_filt_path, index_col='Unnamed: 0')).iloc[:,5:].mean(axis = 1).sort_values()).iloc[:10].index]


# In[56]:


avg_feats = avg_feats['Feature_Name'].values.tolist()


# In[58]:


corr_feats


# In[59]:


chi_feats


# In[60]:


mi_feats


# In[61]:


avg_feats


# In[66]:


(pd.read_csv(feat_filt_path, index_col='Unnamed: 0')).sort_values(by = 'tot_mean', ascending = False)['Feature_Name'].iloc[:10].tolist()


# In[46]:


def set_pipe(clf, features, filename = 'Untitled'):
    piped_clf = make_pipeline(
        (ColumnSelector(cols = features)),
        (SMOTE()),
        (clf)
    )
    piped_clf.fit(X_train,y_train)
    y_pred = piped_clf.predict(X_test)
    con_mat = confusion_matrix(y_test, y_pred)
    avg_f1 = (model_selection.cross_val_score(piped_clf, X_train, y_train, cv = 5, scoring = 'f1')).mean()
    
    print("Cross Val acc score:         ", (model_selection.cross_val_score(piped_clf, X_train, y_train, cv = 5,)).mean())
    print("Cross Val f1  score:         ", avg_f1)
    print()
    print("Overall Acc score:           ", accuracy_score(y_true=y_test, y_pred=y_pred))
    print("Recall score (Tru Pos Rate): ", recall_score(y_true=y_test, y_pred=y_pred))
    print("Precision score:             ", precision_score(y_true=y_test, y_pred=y_pred))
    print("Neg Predictive Val:          ", con_mat[0][0] / (con_mat[0][1] + con_mat[0][0]))
    print("Tru Neg Rate(Specifi):       ", con_mat[0][0] / (con_mat[1][0] + con_mat[0][0]))
    print("F1 score:                    ", f1_score(y_true=y_test, y_pred=y_pred))
    print("Auc score:                   ", roc_auc_score(y_true=y_test, y_score=y_pred))
    print(con_mat)
    print()
    (pd.DataFrame(y_pred)).to_csv(filename + 'y_pred_filt_avg.csv')
    return piped_clf, avg_f1


# In[47]:


list_of_cv_acc = []


# In[48]:


clf1_pipe,clf1_avg_f1 = set_pipe(clf1, avg_feats, 'logit_')
list_of_cv_acc.append(clf1_avg_f1)
clf2_pipe,clf2_avg_f1 = set_pipe(clf2, avg_feats, 'percept_')
list_of_cv_acc.append(clf2_avg_f1)
clf3_pipe,clf3_avg_f1 = set_pipe(clf3, avg_feats, 'NB_')
list_of_cv_acc.append(clf3_avg_f1)
clf4_pipe,clf4_avg_f1 = set_pipe(clf4a, avg_feats, 'lin_svm_')
list_of_cv_acc.append(clf4_avg_f1)


# In[49]:


clf5_pipe,clf5_avg_f1 = set_pipe(clf5, avg_feats, 'c45_')
list_of_cv_acc.append(clf5_avg_f1)


# In[50]:


clf6_pipe,clf6_avg_f1 = set_pipe(clf6, mi_feats, 'knn_')
list_of_cv_acc.append(clf6_avg_f1)


# In[51]:


enclf = EnsembleVoteClassifier((clf1_pipe,clf2_pipe,clf3_pipe,clf4_pipe,clf5_pipe, clf6_pipe), refit = False)
enclf.fit(X_train, y_train)
y_pred = enclf.predict(X_test)
con_mat = confusion_matrix(y_test, y_pred)
    
#print("Cross Val acc score:         ", (model_selection.cross_val_score(enclf, X_train, y_train, cv = 5,)).mean())
#print("Cross Val f1  score:         ", (model_selection.cross_val_score(enclf, X_train, y_train, cv = 5, scoring = 'f1')).mean())
print()
print("Overall Acc score:           ", accuracy_score(y_test, y_pred))
print("Recall score (Tru Pos Rate): ", recall_score(y_test, y_pred))
print("Precision score:             ", precision_score(y_test, y_pred))
print("Neg Predictive Val:          ", con_mat[0][0] / (con_mat[0][1] + con_mat[0][0]))
print("Tru Neg Rate(Specifi):       ", con_mat[0][0] / (con_mat[1][0] + con_mat[0][0]))
print("F1 score:                    ", f1_score(y_test, y_pred))
print("Auc score:                   ", roc_auc_score(y_test, y_pred))
print(con_mat)
print()
(pd.DataFrame(y_pred)).to_csv('maj_vote' + 'y_pred_avg_filt.csv')


# In[52]:


arr = np.array(list_of_cv_acc)
weights = (arr / ((arr).sum()))
weights = list(weights)


# In[53]:


wenclf = EnsembleVoteClassifier((clf1_pipe,clf2_pipe,clf3_pipe,clf4_pipe,clf5_pipe, clf6_pipe), refit = False, weights=weights)
wenclf.fit(X_train, y_train)
y_pred = wenclf.predict(X_test)
print()
print("Overall Acc score:           ", accuracy_score(y_true=y_test, y_pred=y_pred))
print("Recall score (Tru Pos Rate): ", recall_score(y_true=y_test, y_pred=y_pred))
print("Precision score:             ", precision_score(y_true=y_test, y_pred=y_pred))
print("Neg Predictive Val:          ", con_mat[0][0] / (con_mat[0][1] + con_mat[0][0]))
print("Tru Neg Rate(Specifi):       ", con_mat[0][0] / (con_mat[1][0] + con_mat[0][0]))
print("F1 score:                    ", f1_score(y_true=y_test, y_pred=y_pred))
print("Auc score:                   ", roc_auc_score(y_true=y_test, y_score=y_pred))
print(con_mat)
print()
(pd.DataFrame(y_pred)).to_csv('w_maj_vote' + 'y_pred_avg_filt.csv')


# In[ ]:





# In[ ]:




