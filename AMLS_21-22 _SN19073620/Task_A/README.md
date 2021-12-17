# AMLS_21-22-_SN19073620
TASK A:

1. Introduction: 
We use AdaBoost and SVM in Task A for binary classification (no tumor, tumor). 5-fold cross-validation is used here. 

2. This package contains (organization): 
Data_Preprocessing.py, (run first)  NOTE: if you want to preprocess data by yourself, you must put dataset folder
(located at Task B folder) in Task A folder, otherwise, the code will not find the data.
Adaboost.py,(run second) training AdaBoost model.
SVM.py,(run second) training SVM model.
plotter.py (run third) plot fig and analysis.
*.csv file related to preprocessed data and experimental data.

3. The role of each file:
*.py file, python code file which has been introduced before.
NOTE: All images has been preprocessed, you can find them in Task A folder, and they are named as: 
x_test_pca_i.csv, x_train_pca_i.csv, y_test_i.csv, y_train_i.csv,  (i = [1,5]),  for 5-fold cross-validation.
x_test_pca_formal.csv, x_train_pca_formal.scv,  y_test_formal.csv, y_train_formal.csv,  for formal training and test.
The experimental data are:
ada_score_lr=x.csv  indicates the average validation accuracy of AdaBoost model at each num of estimators when lr = x, max_depth =1,
x = {0.2, 0.6, 1, 1.5}.
ada_score_lr=1_dep=2.csv indicates the average validation accuracy of AdaBoost model at each num of estimators when lr = 1, max_depth =2.
grid_search.csv  indicates the average validation accuracy with each pair of [C, gamma], during the grid research.


4. How to run this code:
actually, you don't need run Data_Preprocessing.py as all images has been preprocessed. Also, all experimental data has been saved.
So, you can run Adaboost.py, SVM.py and plotter.py alone directly (no need to run in order anymore). 
NOTE: All codes are run in pycharm, when are you running these code, plese open the Scientific Mode (in view menu), 
so that each code blocks (cells) will appear. Please run only one code block at one time by pressing "ctrl + enter".

5. Necessary packages or header files:
numpy, sklearn, pandas, matplotlib, PIL, glob, gc.










