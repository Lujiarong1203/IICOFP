# IICOFP
 This repository is the data and code for the article "IICOFP: An Interpretable ICO Fraud Prediction Model by Fusing Multi-source Heterogeneous Data".
# IDE

pycharm and jupternotebook, Compiler Environment: python 3.9
# Primary dependency libs or packages:
python3.9 <br>numpy 1.21.5 <br> pandas 1.4.4 <br>seaborn 0.11.2 <br>matplotlib 3.5.2<br>scikit-learn 1.0.2 <br>LightGBM 3.3.4  <br>scikitplot 0.3.7  <br>shap 0.41.0 
# data
The dataset includes the raw dataset (Data Set.csv), the initial ICO fusion dataset constructed after data type conversion and feature filtering (data_1), the dataset after missing value filling (data_2 is), the dataset after One-Hot encoding (data_3), the dataset after outlier handling, the training set after feature selection and category imbalance processing ( data_train, 70%), and test set (data_test, 30%).
# Code 1-4: Data pre-processing codes.
# Code 5-6: Codes for feature contribution analysis, model building, and model comparison.
# Code 7-8: Codes for learning ability and generalization ability analysis of the model.
# Code 9: Model interpretability analysis
# The other 3 Codes are plotting feature statistical distributions, confusion matrices, and feature word clouds
