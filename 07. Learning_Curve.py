import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import validation_curve
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

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

# K-fold
kf=KFold(n_splits=5, shuffle=True, random_state=random_seed)
cnt=1
for train_index, test_index in kf.split(X_train, y_train):
    print(f'Fold:{cnt}, Train set: {len(train_index)}, Test set:{len(test_index)}')
    cnt += 1

# LR
LR=LogisticRegression(random_state=random_seed)

# SVM
SVM=SVC(random_state=random_seed)

# RF
RF=RandomForestClassifier(random_state=random_seed)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)

# LGBM
LGBM=LGBMClassifier(random_state=random_seed)

# XGboost
XG=XGBClassifier(random_state=random_seed)

# 1_SVM
skplt.estimators.plot_learning_curve(SVM, X_train, y_train, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=17)
plt.xlabel('Training sample size', fontsize=17)
plt.ylabel('Score', fontsize=17)
plt.xticks(fontproperties='Times New Roman', fontsize=17)
plt.yticks(fontproperties='Times New Roman', fontsize=17)
plt.title('(a) SVM', y=-0.2, fontproperties='Times New Roman', fontsize=17)
plt.tight_layout()
plt.show()

# 2_LightGBM
skplt.estimators.plot_learning_curve(LGBM, X_train, y_train, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=17)
plt.xlabel('Training sample size', fontsize=17)
plt.ylabel('Score', fontsize=17)
plt.xticks(fontproperties='Times New Roman', fontsize=17)
plt.yticks(fontproperties='Times New Roman', fontsize=17)
plt.title('(b) LightGBM', y=-0.2, fontproperties='Times New Roman', fontsize=17)
plt.tight_layout()
plt.show()

# 3_Adaboost
skplt.estimators.plot_learning_curve(Ada, X_train, y_train, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=17)
plt.xlabel('Training sample size', fontsize=17)
plt.ylabel('Score', fontsize=17)
plt.xticks(fontproperties='Times New Roman', fontsize=17)
plt.yticks(fontproperties='Times New Roman', fontsize=17)
plt.title('(c) Adaboost', y=-0.2, fontproperties='Times New Roman', fontsize=17)
plt.tight_layout()
plt.show()

# 4_GBDT
skplt.estimators.plot_learning_curve(GBDT, X_train, y_train, title=None, cv=kf, random_state=random_seed, shuffle=True, train_sizes=np.linspace(.1, 1.0, 10), scoring="accuracy")
plt.legend(loc='lower right', fontsize=17)
plt.xlabel('Training sample size', fontsize=17)
plt.ylabel('Score', fontsize=17)
plt.xticks(fontproperties='Times New Roman', fontsize=17)
plt.yticks(fontproperties='Times New Roman', fontsize=17)
plt.title('(d) GBDT', y=-0.2, fontproperties='Times New Roman', fontsize=17)
plt.tight_layout()
plt.show()