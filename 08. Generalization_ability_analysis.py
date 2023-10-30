import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import validation_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neural_network import MLPClassifier
import scikitplot as skplt

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# Read data
data_train=pd.read_csv('data/data_train.csv')
data_test=pd.read_csv('data/data_test.csv')
print(data_train.shape, '\n', data_test.shape)

X_train=data_train.drop('Goal', axis=1)
y_train=data_train['Goal']
X_test=data_test.drop('Goal', axis=1)
y_test=data_test['Goal']
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print('测试集中标签数：', Counter(y_test))

random_seed=11

# Comparison of each model
LR=LogisticRegression(random_state=random_seed)
LR.fit(X_train, y_train)
y_pred_LR=LR.predict(X_test)
y_proba_LR=LR.predict_proba(X_test)
cm_LR=confusion_matrix(y_test, y_pred_LR)
print(cm_LR)

# LightGBM
lgbm=LGBMClassifier(random_state=random_seed)
lgbm.fit(X_train, y_train)
y_pred_lgbm=lgbm.predict(X_test)
y_proba_lgbm=lgbm.predict_proba(X_test)
cm_lgbm=confusion_matrix(y_test, y_pred_lgbm)
print(cm_lgbm)
#
# XGboost
xg=XGBClassifier(random_state=random_seed)
xg.fit(X_train, y_train)
y_pred_xg=xg.predict(X_test)
y_proba_xg=xg.predict_proba(X_test)
cm_xg=confusion_matrix(y_test, y_pred_xg)
print(cm_xg)

# KNN
knn=KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn=knn.predict(X_test)
y_proba_knn=knn.predict_proba(X_test)
cm_knn=confusion_matrix(y_test, y_pred_knn)
print(cm_knn)

# SVM
SVM=SVC(random_state=random_seed, probability=True)
SVM.fit(X_train, y_train)
y_pred_SVM=SVM.predict(X_test)
y_proba_SVM=SVM.predict_proba(X_test)
cm_SVM=confusion_matrix(y_test, y_pred_SVM)
print(cm_SVM)

# PA
# PA=PassiveAggressiveClassifier(random_state=random_seed)
# PA.fit(x_train, y_train)
# y_pred_PA=PA.predict(x_test)
# y_proba_PA=PA.predict_proba(x_test)
# cm_PA=confusion_matrix(y_test, y_pred_PA)

# ET
ETC=ExtraTreesClassifier(random_state=random_seed)
ETC.fit(X_train, y_train)
y_pred_ETC=ETC.predict(X_test)
y_proba_ETC=ETC.predict_proba(X_test)
cm_ETC=confusion_matrix(y_test, y_pred_ETC)
print(cm_ETC)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)
Ada.fit(X_train, y_train)
y_pred_Ada=Ada.predict(X_test)
y_proba_Ada=Ada.predict_proba(X_test)
cm_Ada=confusion_matrix(y_test, y_pred_Ada)
print(cm_Ada)

#SGD
SGD=SGDClassifier(random_state=random_seed, loss="log")
SGD.fit(X_train, y_train)
y_pred_SGD=SGD.predict(X_test)
y_proba_SGD=SGD.predict_proba(X_test)
cm_SGD=confusion_matrix(y_test, y_pred_SGD)
print(cm_SGD)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)
GBDT.fit(X_train, y_train)
y_pred_GBDT=GBDT.predict(X_test)
y_proba_GBDT=GBDT.predict_proba(X_test)
cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)
print(cm_GBDT)

# DT
DT=DecisionTreeClassifier(random_state=random_seed)
DT.fit(X_train, y_train)
y_pred_DT=DT.predict(X_test)
y_proba_DT=DT.predict_proba(X_test)
cm_DT=confusion_matrix(y_test, y_pred_DT)
print(cm_DT)

# MLP
MLP=MLPClassifier(random_state=random_seed)
MLP.fit(X_train, y_train)
y_pred_MLP=MLP.predict(X_test)
y_proba_MLP=MLP.predict_proba(X_test)
cm_MLP=confusion_matrix(y_test, y_pred_MLP)
print(cm_MLP)

# RF
RF=RandomForestClassifier(random_state=random_seed)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)
y_proba_RF = RF.predict_proba(X_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)
print(cm_RF)


# # KS curve

# GBDT
skplt.metrics.plot_ks_statistic(y_test, y_proba_GBDT, title=None, text_fontsize=15, figsize=(6, 6))
# plt.title('(a) IICOFDM', y=-0.2, fontsize=15)
plt.legend(fontsize=15, loc='lower right')
plt.savefig('Fig.8(b).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# # cumulative_gain curve
# GBDT
skplt.metrics.plot_cumulative_gain(y_test, y_proba_GBDT, title=None, text_fontsize=15, figsize=(6, 6))
plt.legend(loc='lower right', fontsize=15)
plt.savefig('Fig.8(c).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# Lift curve
skplt.metrics.plot_lift_curve(y_test, y_proba_GBDT, title=None, text_fontsize=15, figsize=(6, 6))
plt.legend(loc='upper right', fontsize=15)
plt.savefig('Fig.8(d).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
plt.show()

# # Pre_Rec_Curve
# skplt.metrics.plot_precision_recall(y_test, y_proba_GBDT, title=None, text_fontsize=15, figsize=(6, 6))
# plt.legend(loc='lower right', fontsize=15)
# plt.savefig('Fig.10(d).jpg', dpi=600, bbox_inches='tight',pad_inches=0)
# plt.show()

# 多个模型的ROC曲线对比
# import matplotlib.pylab as plt
plt.rc('font',family='Times New Roman')
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.sans-serif']=['SimHei']
fpr1, tpr1, thres1 = roc_curve(y_test, y_proba_LR[:, 1])
fpr2, tpr2, thres2 = roc_curve(y_test, y_proba_SVM[:, 1])
fpr3, tpr3, thres3 = roc_curve(y_test, y_proba_knn[:,1])
fpr4, tpr4, thres4 = roc_curve(y_test, y_proba_DT[:, 1])
fpr5, tpr5, thres5 = roc_curve(y_test, y_proba_RF[:, 1])
fpr6, tpr6, thres6 = roc_curve(y_test, y_proba_ETC[:, 1])
fpr7, tpr7, thres7 = roc_curve(y_test, y_proba_xg[:, 1])
fpr8, tpr8, thres8 = roc_curve(y_test, y_proba_Ada[:, 1])
fpr9, tpr9, thres9 = roc_curve(y_test, y_proba_lgbm[:, 1])
fpr10, tpr10, thres10 = roc_curve(y_test, y_proba_GBDT[:, 1])


plt.figure(figsize=(6, 6))
plt.grid()
plt.plot(fpr1, tpr1, 'b', label='LR ', color='k',lw=1.5,ls='--')
plt.plot(fpr2, tpr2, 'b', label='SVM ', color='darkorange',lw=1.5,ls='--')
plt.plot(fpr3, tpr3, 'b', label='KNN ', color='peru',lw=1.5,ls='--')
plt.plot(fpr4, tpr4, 'b', label='DT ', color='lime',lw=1.5,ls='--')
plt.plot(fpr5, tpr5, 'b', label='RF ', color='fuchsia',lw=1.5,ls='--')

plt.plot(fpr6, tpr6, 'b', label='ETC ', color='cyan',lw=1.5,ls='--')
plt.plot(fpr7, tpr7, 'b', label='XGboost ', color='green',lw=1.5,ls='--')
plt.plot(fpr8, tpr8, 'b', label='Adaboost ', color='blue',lw=1.5,ls='--')
plt.plot(fpr9, tpr9, 'b', label='LightGBM ', color='violet',lw=1.5, ls='--')
plt.plot(fpr10, tpr10, 'b', ms=1,label='GBDT ', lw=3.5,color='red',marker='*')

plt.plot([0, 1], [0, 1], 'darkgrey')
plt.xlim([-0.02, 1.02])
plt.ylim([-0.02, 1.02])
plt.ylabel('True Positive Rate', fontname='Times New Roman', fontsize=15)
plt.xlabel('False Positive Rate', fontname='Times New Roman', fontsize=15)
plt.tick_params(labelsize=12)
plt.legend(fontsize=15)
plt.savefig('Fig.8(a).jpg', dpi=700, bbox_inches='tight',pad_inches=0)
plt.show()
