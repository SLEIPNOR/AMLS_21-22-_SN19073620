# AMLS_21-22-_SN19073620
TASK A:

Introduction: We use AdaBoost and SVM in Task A for binary classification (no tumor, tumor). 5-fold cross-validation is 
used here. 

This package contains: 
1. Data_Preprocessing.py, (run first)  NOTE: if you want to preprocess data by yourself, you must put dataset folder
(located at Task B folder) in Task A folder, otherwise, the code will not find the data.
2. AdaBoost.py,(run second) training AdaBoost model.
3. SVM.py,(run second) training SVM model.
4. plotter.py (run third) plot fig and analysis.
5. csv file related to preprocessed data and experimental data

The role of each file
NOTE: All images has been preprocessed, you can find them in Task A folder, and they are named as: 
x_test_pca_i.csv, x_train_pca_i.csv, y_test_i.csv, y_train_i.csv,  (i = [1,5]),  for 5-fold cross-validation
x_train_pca_formal.csv, y_test_formal.csv,   for formal training and test
The experimental data are:
ada_score_lr=x.csv  indicates the average validation accuracy of AdaBoost model at each num of estimators when lr = x, max_depth =1.
x = {0.2, 0.6, 1, 1.5}
ada_score_lr=1_dep=2.csv indicates the average validation accuracy of AdaBoost model at each num of estimators when lr = 1, max_depth =2.



How to run this code:
actually, you don't need run Data_Preprocessing.py as all images has been preprocessed. Also, all experimental data has been saved.
So, you can run AdaBoost.py, SVM.py and plotter.py alone directly (no need to run in order anymore). 
NOTE: All codes are run in pycharm, when you running these code, plese open the Scientific Mode (in view menu), 
so that each code blocks (cells) will appear. Please run only one code block at one time by press "ctrl + enter".
