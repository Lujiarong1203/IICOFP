import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
import xgboost
from lightgbm import LGBMClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, roc_auc_score, roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,accuracy_score,roc_curve

import scikitplot as skplt

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

# Read data
data_train=pd.read_csv('data/data_train')
data_test=pd.read_csv('data/data_test')
print(data_train.shape, '\n', data_test.shape)

X_train=data_train.drop('Goal', axis=1)
y_train=data_train['Goal']
X_test=data_test.drop('Goal', axis=1)
y_test=data_test['Goal']
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print('测试集中标签数：', Counter(y_test))
random_seed=11

#
# Comparison of each model
LR=LogisticRegression(random_state=random_seed)
LR.fit(X_train, y_train)
y_pred_LR=LR.predict(X_test)
y_proba_LR=LR.predict_proba(X_test)
cm_LR=confusion_matrix(y_test, y_pred_LR)
#
# LightGBM
lgbm=LGBMClassifier(random_state=random_seed)
lgbm.fit(X_train, y_train)
y_pred_lgbm=lgbm.predict(X_test)
y_proba_lgbm=lgbm.predict_proba(X_test)
cm_lgbm=confusion_matrix(y_test, y_pred_lgbm)

# XGboost
xg=xgboost.XGBClassifier(random_state=random_seed)
xg.fit(X_train, y_train)
y_pred_xg=xg.predict(X_test)
y_proba_xg=xg.predict_proba(X_test)
cm_xg=confusion_matrix(y_test, y_pred_xg)

# KNN
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn=knn.predict(X_test)
y_proba_knn=knn.predict_proba(X_test)
cm_knn=confusion_matrix(y_test, y_pred_knn)

# SVM
SVM=SVC(random_state=random_seed, probability=True)
SVM.fit(X_train, y_train)
y_pred_SVM=SVM.predict(X_test)
y_proba_SVM=SVM.predict_proba(X_test)
cm_SVM=confusion_matrix(y_test, y_pred_SVM)

# # PA
# # PA=PassiveAggressiveClassifier(random_state=random_seed)
# # PA.fit(x_train, y_train)
# # y_pred_PA=PA.predict(x_test)
# # y_proba_PA=PA.predict_proba(x_test)
# # cm_PA=confusion_matrix(y_test, y_pred_PA)
#
# ET
ETC=ExtraTreesClassifier(random_state=random_seed)
ETC.fit(X_train, y_train)
y_pred_ETC=ETC.predict(X_test)
y_proba_ETC=ETC.predict_proba(X_test)
cm_ETC=confusion_matrix(y_test, y_pred_ETC)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)
Ada.fit(X_train, y_train)
y_pred_Ada=Ada.predict(X_test)
y_proba_Ada=Ada.predict_proba(X_test)
cm_Ada=confusion_matrix(y_test, y_pred_Ada)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)
GBDT.fit(X_train, y_train)
y_pred_GBDT=GBDT.predict(X_test)
y_proba_GBDT=GBDT.predict_proba(X_test)
cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)

# DT
DT=DecisionTreeClassifier(random_state=random_seed)
DT.fit(X_train, y_train)
y_pred_DT=DT.predict(X_test)
y_proba_DT=DT.predict_proba(X_test)
cm_DT=confusion_matrix(y_test, y_pred_DT)

# MLP
MLP=MLPClassifier(random_state=random_seed)
MLP.fit(X_train, y_train)
y_pred_MLP=MLP.predict(X_test)
y_proba_MLP=MLP.predict_proba(X_test)
cm_MLP=confusion_matrix(y_test, y_pred_MLP)

# RF
RF=RandomForestClassifier(random_state=random_seed)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)
y_proba_RF = RF.predict_proba(X_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)

# # confusion matrix heat map
# DT
skplt.metrics.plot_confusion_matrix(y_test, y_pred_DT, title=None, cmap='tab20', text_fontsize=15)
plt.title('(a) DT', y=-0.2, fontsize=15)
plt.savefig('Fig.7(c).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# RF
skplt.metrics.plot_confusion_matrix(y_test, y_pred_RF, title=None, cmap='tab20', text_fontsize=15)
plt.title('(b) RF', y=-0.2, fontsize=15)
plt.savefig('Fig.7(f).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# XGboost
skplt.metrics.plot_confusion_matrix(y_test, y_pred_xg, title=None, cmap='tab20', text_fontsize=15)
plt.title('(c) XGboost', y=-0.2, fontsize=15)
plt.savefig('Fig.7(d).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# GBDT
skplt.metrics.plot_confusion_matrix(y_test, y_pred_GBDT, title=None, cmap='tab20', text_fontsize=15)
plt.title('(d) GBDT', y=-0.2, fontsize=15)
plt.savefig('Fig.7(f).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()