# Readmitted-Rate

## Introduction

**_Patients readmitted rate_**, which is a **supervised learning** classification problem.

Using a combination of models **(KNN, SVM, Decision Tree, Perceptron and Naïve Bayes algorithms）**
to deal with the real-world datasets from UCI machine learning lab.

## Including:
EDA<br>
Data Cleaning<br>
Feature Seletion<br>
Models tuning<br>
Ensemble<br>
and get the best accuracy about the readmitted rate.


## Implementation:
 
Put these file 
{
Smote_BackwardWrapper_Pipeline.py, 

ForwardWrapper_Smote.py,

Results for the test set.py,

GridSearch Main Version Updated.py,

Base_Scoring_Algorithm (Corr, Chi2, Mutual_Info, C4_5 Importance).py,

New_train_set.csv,

New_test_set.csv 
} 
in the same location. 

## Filter Method:
Base_Scoring_Algorithm 
### (Corr, Chi2, Mutual_Info, C4_5 Importance).py is for feature selection:
use to calculate correlation, chi-square, mutual information between variables and target label

### GridSearch Main Version Updated.py:
Using the results of Scoring to do feature selection with filter method;
Tuning different models with selected features using gridsearch

###Results for the test set.py:
Compute different accuracy score of test set with best tuned parameters 

## Wrapper Method:
Fordward wrapper to find the global best collection of features.

### ForwardWrapper_Smote.py:
Input training set data and return selected features as csv

### Smote_BackwardWrapper_Pipeline.py
Take advantages of pipeline to do SMOTE, backward wrapper and tuning;

Compute different accuracy score of test set with best tuned parameters;

Save best parameters, test scores, and feature selected as csv






	
	