README FOR FINAL PROJECT CODE
============

This is the README file for the final project python programs 
 
Put these file {
Smote_BackwardWrapper_Pipeline.py, 
ForwardWrapper_Smote.py,
Results for the test set.py,
GridSearch Main Version Updated.py,
Base_Scoring_Algorithm (Corr, Chi2, Mutual_Info, C4_5 Importance).py,
New_train_set.csv,
New_test_set.csv 
} in the same location. 

Filter Method:
Base_Scoring_Algorithm (Corr, Chi2, Mutual_Info, C4_5 Importance).py is for feature selection:
	use to calculate correlation, chi-square, mutual information between variables and target label

GridSearch Main Version Updated.py:
	use the results of Scoring to do feature selection with filter method;
	tune different models with selected features using gridsearch

Results for the test set.py:
	compute different accuracy score of test set with best tuned parameters 

Wrapper Method:
	Fordward wrapper to find the global best collection of features.

ForwardWrapper_Smote.py:
	input training set data and return selected features as csv

Smote_BackwardWrapper_Pipeline.py,:
	take advantages of pipeline to do SMOTE, backward wrapper and tuning;
	compute different accuracy score of test set with best tuned parameters;
	save best parameters, test scores, and feature selected as csv

============
Thank you for running our code! Thx




	
	