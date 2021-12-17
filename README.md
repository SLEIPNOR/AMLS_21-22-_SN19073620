# AMLS_21-22-_SN19073620
TASK B:

1. Introduction: 
We use 24 layers depth (24LD) ResNet model with different hyperparameters in Task B for multiclass classification 
(no tumor, meningioma tumor, glioma tumor, pituitary tumor). 10-fold cross-validation is used here. 

2. This package contains (organization): 
basic_experts_model.py (24LD model).
ResBlock.py (one ResNet block).
data_augmentation.py (data augmentation).
data_reading.py (load training and test data).
cross_validation_training.py (train model with 10-fold cross-validation) .
model_testing.py (load model and test its performance).
Plotter.py (plot the experimental result).
dataset folder contains training dataset and test dataset.
checkpoints_modelA folder for model A save.
checkpoints_modelB folder for model B save.
*.csv file related to experimental data.

3. The role of each file:
*.py file, python code file which has been introduced before.
The experimental data are:
all_acc_1e-x.csv indicates the accurcy of model with dropout and different regularization penalty x ={1,2,3,4}.
all_loss_1e-x.csv indicates the loss of model with dropout and different regularization penalty x ={1,2,3,4}.
all_scores_1e-x.csv indicates the validation accuracy of each fold in one cross-validation with dropout and different regularization penalty x ={1,2,3,4}.
dropout_acc.csv, dropout_loss.csv, dropout_scores.csv is mearsured for model only with dropout which is same as above.
no_acc.csv, no_loss.csv, no_scores.csv  is mearsured for model without both dropout and regularization which is also same as above.
reg_acc.csv, reg_loss.csv, reg_scores.csv is mearsured for model only with regularization penalty = 1e-3, which is also same as above.
modelA_aug_acc.csv, modelA_aug_loss.csv, modelA_aug_scores.csv is measured for model A after data augmentation, which is same as above.
modelB_aug_acc.csv, modelB_aug_loss.csv, modelB_aug_scores.csv is measured for model B after data augmentation, which is same as above.
reg=x.csv indicates the history of dynamic learning rate search of model with only regularization peanlty, where peanlty x = {0, 1e-1, 1e-2, 1e-3, 1e-4}.
dropout_only.csv indicates the history of dynamic learning rate search of model with only dropout.  
checkpoints_modelA and checkpoints_modelB are the saving models of model A and B.

4. How to run this code:
Just run the file "cross_validation_training.py" to find the best learning rate and train an ensemble model with the best hyperparameters,
Training model may spend much time, so you if you don't want to wait so much time for training, you can also use the saving model directly.
For how to use: run the file "model_testing.py" to load and test the saving model.
NOTE: All codes are run in pycharm, when you are running these code, plese open the Scientific Mode (in view menu), 
so that each code blocks (cells) will appear. Please run only one code block at one time by pressing "ctrl + enter".

5. Necessary packages or header files:
Tensorflow 2.0.0, GTX/RTX Grahpic Cards, Cuda 10.0, Cudnn 7.6.5
keras, numpy, sklearn, pandas, matplotlib, PIL, glob, gc, random.uniform, collections.Counter, ipykernel.





