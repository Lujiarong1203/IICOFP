import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from imbalanced_ensemble.utils._plot import plot_2Dprojection_and_cardinality
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN


from sklearn.linear_model import LassoCV
from sklearn.feature_selection import mutual_info_classif as MIC
from sklearn.feature_selection import SelectKBest, RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data=pd.read_csv('data/data_4.csv')
print(data.head(5), '\n', data.shape)

X=data.drop('Goal', axis=1)
y=data['Goal']

print(X.shape, y.shape)

random_seed=11
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_seed, stratify=y)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(Counter(y_train))

"""
SMOTE采样
"""
X_train_SMOTE, y_train_SMOTE =TomekLinks().fit_resample(X_train, y_train)
print('SMOTE_train_set:', Counter(y_train_SMOTE), '\n', 'test_set:', Counter(y_test))

# Lasso特征选择
#调用LassoCV函数，并进行交叉验证，默认cv=3
lasso_model = LassoCV(alphas = [0.1,1,0.001,0.0005],random_state=random_seed).fit(X_train_SMOTE,y_train_SMOTE)
print(lasso_model.alpha_) #模型所选择的最优正则化参数alpha

#输出看模型最终选择了几个特征向量，剔除了几个特征向量
coef = pd.Series(lasso_model.coef_, index = X_train_SMOTE.columns)
print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

# 索引和重要性做成dataframe形式
FI_lasso = pd.DataFrame({"Feature Importance":lasso_model.coef_}, index=X_train_SMOTE.columns)

# 由高到低进行排序
FI_lasso.sort_values("Feature Importance",ascending=False).round(3)
# print(FI_lasso)

# 获取重要程度大于0的系数指标
FI_lasso[FI_lasso["Feature Importance"] !=0 ].sort_values("Feature Importance").plot(kind="barh",color='cornflowerblue',alpha=0.8)
plt.xticks(rotation=0)
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature coefficient',fontsize=11)
plt.ylabel('Feature name',fontsize=11)
plt.tick_params(labelsize = 11)
plt.savefig('Fig.6.jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()

# 输出Lasso特征选择后的 训练集 和 测试集
drop_colums=coef.index[abs(coef.values)==0]
X_train_SMOTE_lasso=X_train_SMOTE.drop(drop_colums, axis=1)
X_test_lasso=X_test.drop(drop_colums, axis=1)
print('Lasso特征选择后的训练集和特征集的维度', X_train_SMOTE_lasso.shape, X_test_lasso.shape)


# MIC特征选择
k_best = 50
mic_model=SelectKBest(MIC, k=k_best)
X_mic = mic_model.fit_transform(X_train_SMOTE, y_train_SMOTE)
mic_scores=mic_model.scores_
mic_indices=np.argsort(mic_scores)[::-1]
mic_k_best_features = list(X_train_SMOTE.columns.values[mic_indices[0:k_best]])
FI_mic = pd.DataFrame({"Feature Importance":mic_scores}, index=X_train_SMOTE.columns)
FI_mic[FI_mic["Feature Importance"] !=0 ].sort_values("Feature Importance").plot(kind="barh",color='firebrick',alpha=0.8)
plt.xticks(rotation=0,fontsize=11)
plt.xlabel('特征重要程度',fontsize=11)
plt.ylabel('特征名称',fontsize=11)
plt.show()

# MIC特征选择后的 训练集 和 测试集
X_train_SMOTE_MIC=X_train_SMOTE[mic_k_best_features]
X_test_MIC=X_test[mic_k_best_features]
print('MIC特征选择后的训练集和特征集的维度', X_train_SMOTE_MIC.shape, X_test_MIC.shape)


# RFE特征选择
rfe_model = RFE(RandomForestClassifier(random_state=random_seed))
rfe = rfe_model.fit(X_train_SMOTE, y_train_SMOTE)
# X_train_rfe = rfe.transform(X_train)# 最优特征

# feature_ranking = rfe.ranking_
# print(feature_ranking)
feature_importance_values = rfe.estimator_.feature_importances_
# print(feature_importance_values)

# # RFE特征选择后的 训练集 和 测试集
X_train_SMOTE_RFE= X_train_SMOTE[rfe.get_feature_names_out()]
X_test_RFE=X_test[rfe.get_feature_names_out()]
print('RFE特征选择后的训练集和特征集的维度', X_train_SMOTE_RFE.shape, X_test_RFE.shape)

FI_RFE=pd.DataFrame(feature_importance_values, index=X_train_SMOTE_RFE.columns, columns=['features importance'])
print(FI_RFE)

## 由高到低进行排序
FI_RFE=FI_RFE.sort_values("features importance",ascending=False).round(3)
# print(FI_RFE)

# 获取重要程度大于0的系数指标
# plt.figure(figsize=(15, 10))
FI_RFE[FI_RFE["features importance"] !=0 ].sort_values("features importance").plot(kind="barh",color='firebrick',alpha=0.8)
plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=30)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=15)
plt.ylabel('Feature name',fontsize=15)
plt.tick_params(labelsize = 11)
plt.savefig('Fig.5(d).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()

"""
比较不同方法组合下模型的性能
"""
def model_comparison(x_train_set, y_train_set, x_test_set, y_test_set, estimator):
    # 归一化
    # mm = StandardScaler()
    # x_train_std = pd.DataFrame(mm.fit_transform(x_train_set))
    # x_train_std.columns=x_train_set.columns
    # x_test_std=pd.DataFrame(mm.fit_transform(x_test_set))
    # x_test_std.columns=x_test_set.columns

    # # estimator
    est = estimator
    est.fit(x_train_set, y_train_set)
    y_pred = est.predict(x_test_set)
    score_data = []
    scoring = [accuracy_score, precision_score, recall_score, f1_score, roc_auc_score]
    for sco in scoring:
        score = sco(y_test_set, y_pred)
        score_data.append(score)
    print(score_data)

# 定义模型
XG=XGBClassifier(random_state=random_seed)
GBDT=GradientBoostingClassifier(random_state=random_seed)
from lightgbm import LGBMClassifier
LGBM=LGBMClassifier(random_state=random_seed)
RF=RandomForestClassifier(random_state=random_seed)
from sklearn.ensemble import AdaBoostClassifier
Ada=AdaBoostClassifier(random_state=random_seed)

# 比较性能
model_comparison(X_train, y_train, X_test, y_test, GBDT)
print("No any process")
model_comparison(X_train_SMOTE_lasso, y_train_SMOTE, X_test_lasso, y_test, GBDT)
print('SMOTE+Lasso')
model_comparison(X_train_SMOTE_MIC, y_train_SMOTE, X_test_MIC, y_test, GBDT)
print("SMOTE+MIC")
model_comparison(X_train_SMOTE_RFE, y_train_SMOTE, X_test_RFE, y_test, GBDT)
print("SMOTE+RFE")
# model_comparison(X_train_SMOTE_RFECV, y_train_SMOTE, X_test_RFECV, y_test, XG)
# print('SMOTE+RFECV')



"""
经过比较后得到，SMOTE和RFE联合预处理后的数据集，模型性能最优，因此保存数据
"""
data_train=pd.concat([X_train_SMOTE_lasso, y_train_SMOTE], axis=1)
data_test=pd.concat([X_test_lasso, y_test], axis=1)
print(data_train.shape, data_test.shape)

data_train.to_csv(path_or_buf=r'data/data_train.csv', index=None)
data_test.to_csv(path_or_buf=r'data/data_test.csv', index=None)

"""
采样前后数据的2D分布
"""
# 采样——特征选择前
mm=StandardScaler()
X_train_std=pd.DataFrame(mm.fit_transform(X_train))
X_train_std.columns=X_train.columns
plot_2Dprojection_and_cardinality(X_train_std, y_train)
plt.tick_params(labelsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.legend(loc='upper right')
plt.xticks([0, 1], ['Fraudulent', 'Successful'], rotation='horizontal')
plt.show()

# 采样——特征选择后
mm=StandardScaler()
X_train_SMOTE_lasso_std=pd.DataFrame(mm.fit_transform(X_train_SMOTE_lasso))
X_train_SMOTE_lasso_std.columns=X_train_SMOTE_lasso.columns
plot_2Dprojection_and_cardinality(X_train_SMOTE_lasso_std, y_train_SMOTE)
plt.tick_params(labelsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
# plt.legend(loc='upper right')
plt.xticks([0, 1], ['Failed', 'Success'], rotation='horizontal')
plt.show()



