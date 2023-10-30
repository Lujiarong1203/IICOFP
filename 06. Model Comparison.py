# 导入包
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sn
from sklearn.metrics import precision_score, accuracy_score,f1_score, recall_score, roc_auc_score, matthews_corrcoef
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 导入模型
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import NuSVC
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
# from catboost import CatBoostClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

random_seed=11

data_train=pd.read_csv('data/data_train.csv')
data_test=pd.read_csv('data/data_test.csv')
print(data_train.shape, '\n', data_test.shape)

X_train=data_train.drop('Goal', axis=1)
y_train=data_train['Goal']
X_test=data_test.drop('Goal', axis=1)
y_test=data_test['Goal']
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# 将模型用k-v的字典存放起来，所有模型的超参数是默认的
classfiers = {'LogisticRegression': LogisticRegression(), 'PassiveAggressiveClassifier': PassiveAggressiveClassifier(),
              'SGDClassifier':SGDClassifier(),'LinearSVC':LinearSVC(),'SVC':SVC(),'KNeighborsClassifier':KNeighborsClassifier(),
              'GaussianNB':GaussianNB(),'BernoulliNB':BernoulliNB(),
              'DecisionTreeClassifier':DecisionTreeClassifier(),'ExtraTreeClassifier':ExtraTreeClassifier(),
              'MLPClassifier':MLPClassifier(),'RandomForestClassifier':RandomForestClassifier(),'ExtraTreesClassifier':ExtraTreesClassifier(),
              'GradientBoostingClassifier':GradientBoostingClassifier(random_state=random_seed),'AdaBoostClassifier':AdaBoostClassifier(),
              'HistGradientBoostingClassifier':HistGradientBoostingClassifier(),
              'RidgeClassifier': RidgeClassifier(), 'LGBMClassifier': LGBMClassifier(), 'XGBClassifier': XGBClassifier()
              }

result_pd = pd.DataFrame()
cls_nameList = []
# 这些性能指标，可以跟进你真实的需求，进行增删。
accuracys=[]
precisions=[]
recalls=[]
F1s=[]
AUCs=[]
MMCs = []

for cls_name, cls in classfiers.items():
    print("start training:", cls_name)
    cls.fit(X_train, y_train)
    y_pred = cls.predict(X_test)
    cls_nameList.append(cls_name)
    accuracys.append(accuracy_score(y_test, y_pred))
    precisions.append(precision_score(y_test, y_pred))
    recalls.append(recall_score(y_test, y_pred))
    F1s.append(f1_score(y_test, y_pred))
    AUCs.append(roc_auc_score(y_test, y_pred))
    MMCs.append(matthews_corrcoef(y_test, y_pred))

result_pd['classfier_name'] = cls_nameList
result_pd['accuracy'] = accuracys
result_pd['precision'] = precisions
result_pd['recall'] = recalls
result_pd['F1'] = F1s
result_pd['AUC'] = AUCs
result_pd['MMC'] = MMCs
print(result_pd)

result_pd.to_csv('./xxxx_result_compare.csv', index=0)

print("work done!")
