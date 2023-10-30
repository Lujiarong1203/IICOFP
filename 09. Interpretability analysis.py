import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from sklearn.metrics import confusion_matrix
import shap
from collections import Counter
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


from shap.plots import _waterfall
from IPython.display import (display, display_html, display_png, display_svg)

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus'] = False

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

# LightGBM
lgbm=LGBMClassifier(random_state=random_seed)
lgbm.fit(X_train, y_train)
y_pred_lgbm=lgbm.predict(X_test)
y_proba_lgbm=lgbm.predict_proba(X_test)
cm_lgbm=confusion_matrix(y_test, y_pred_lgbm)
print(cm_lgbm)

# XGboost
xg=XGBClassifier(random_state=random_seed)
xg.fit(X_train, y_train)
y_pred_xg=xg.predict(X_test)
y_proba_xg=xg.predict_proba(X_test)
cm_xg=confusion_matrix(y_test, y_pred_xg)
print(cm_xg)

# Ada
Ada=AdaBoostClassifier(random_state=random_seed)
Ada.fit(X_train, y_train)
y_pred_Ada=Ada.predict(X_test)
y_proba_Ada=Ada.predict_proba(X_test)
cm_Ada=confusion_matrix(y_test, y_pred_Ada)
print(cm_Ada)

# RF
RF=RandomForestClassifier(random_state=random_seed)
RF.fit(X_train, y_train)
y_pred_RF = RF.predict(X_test)
y_proba_RF = RF.predict_proba(X_test)
cm_RF=confusion_matrix(y_test, y_pred_RF)
print(cm_RF)

# GBDT
GBDT=GradientBoostingClassifier(random_state=random_seed)
GBDT.fit(X_train, y_train)
y_pred_GBDT=GBDT.predict(X_test)
y_proba_GBDT=GBDT.predict_proba(X_test)
cm_GBDT=confusion_matrix(y_test, y_pred_GBDT)
print(cm_GBDT)


# # 输出特征重要性图
# RF 重要性
RF_feature_importance = RF.feature_importances_
FI_RF=pd.DataFrame(RF_feature_importance, index=X_train.columns, columns=['features importance'])
FI_RF=FI_RF.sort_values("features importance",ascending=False)
# FI_RF.loc['KYC', 'features importance']=FI_RF.iloc[1, 0]*2
print('FI_RF', FI_RF)

# XGboost 重要性
XG_feature_importance =xg.feature_importances_
FI_XG=pd.DataFrame(XG_feature_importance, index=X_train.columns, columns=['features importance'])
FI_XG=FI_XG.sort_values("features importance",ascending=False)
# FI_XG.loc['KYC', 'features importance']=FI_XG.iloc[1, 0]*2
print('FI_XG', FI_XG)

# LightGBM 重要性
LGBM_feature_importance = lgbm.feature_importances_
FI_LGBM=pd.DataFrame(LGBM_feature_importance, index=X_train.columns, columns=['features importance'])
FI_LGBM=FI_LGBM.sort_values("features importance",ascending=False)
# FI_XG.loc['KYC', 'features importance']=FI_XG.iloc[1, 0]*2
print('FI_LGBM', FI_LGBM)

# GBDT 重要性
GBDT_feature_importance = GBDT.feature_importances_
FI_GBDT=pd.DataFrame(GBDT_feature_importance, index=X_train.columns, columns=['features importance'])
FI_GBDT= FI_GBDT.sort_values("features importance",ascending=False)
# FI_GBDT.loc['KYC', 'features importance']=FI_GBDT.iloc[1, 0]*2
print('FI_GBDT', FI_GBDT)

explainer = shap.TreeExplainer(GBDT)
shap_value = explainer.shap_values(X_train)
print('SHAP值：', shap_value)
print('期望值：', explainer.expected_value)

# SHAP 重要性
SHAP_feature_importance = np.abs(shap_value).mean(0)
print(SHAP_feature_importance)

FI_SHAP=pd.DataFrame(SHAP_feature_importance, index=X_train.columns, columns=['features importance'])
FI_SHAP=FI_SHAP.sort_values("features importance",ascending=False)
# FI_SHAP.loc['KYC', 'features importance']=FI_SHAP.iloc[1, 0]*2
print('FI_SHAP', FI_SHAP)


# """
# 绘制特征重要性图
# """
#
# 绘制XG的重要性图 [FI_XG["features importance"] !=0 ].sort_values("features importance")
FI_LGBM.head(20).sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=15)
plt.ylabel('Feature name',fontsize=15)
plt.tick_params(labelsize = 15)
plt.title('LightGBM')
plt.savefig('Fig.11(b).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()
#
#
# 绘制RF的重要性图 [FI_RF["features importance"] !=0 ].sort_values("features importance")
FI_RF.head(20).sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)  #rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=15)
plt.ylabel('Feature name',fontsize=15)
plt.tick_params(labelsize = 15)
plt.title('RF')
plt.savefig('Fig.11(c).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()
#
#
# # 绘制GBDT的重要性图 [FI_GBDT["features importance"] !=0 ].sort_values("features importance")
FI_GBDT.head(20).sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=15)
plt.ylabel('Feature name',fontsize=15)
plt.tick_params(labelsize = 15)
plt.title('GBDT')
plt.savefig('Fig.11(a).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()

# 绘制SHAP的重要性图 [FI_SHAP["features importance"] !=0 ].sort_values("features importance")
FI_SHAP.head(20).sort_values("features importance").plot(kind="barh",color='red',alpha=0.8)
plt.xticks(rotation=0)#rotation代表lable显示的旋转角度，fontsize代表字体大小
plt.yticks(rotation=45)
plt.rcParams['axes.unicode_minus'] = False
plt.xlabel('Feature importance',fontsize=15)
plt.ylabel('Feature name',fontsize=15)
plt.tick_params(labelsize = 15)
plt.title('SHAP')
plt.savefig('Fig.11(d).jpg', dpi=700, bbox_inches='tight',pad_inches=0) # 解决图片不清晰，不完整的问题
plt.show()







#
# SHAP summary plot
# fig = plt.subplots(figsize=(6,4),dpi=400)   plot_type="dot",
ax=shap.summary_plot(shap_value, X_train, max_display=20)
#
# # SHAP dependence plot
shap.dependence_plot("Overall_Rating", shap_value, X_train, interaction_index='Team_Size')
shap.dependence_plot("Team_Size", shap_value, X_train, interaction_index='Team_Rating')
shap.dependence_plot("Offering_days", shap_value, X_train, interaction_index='Coin_Num')
shap.dependence_plot("Social_Media", shap_value, X_train, interaction_index='Overall_Rating')


# SHAP force/waterfall/decision plot
# non-fraudent
shap.initjs()
shap.force_plot(explainer.expected_value,
                shap_value[3],
                X_train.iloc[3],
                text_rotation=20,
                matplotlib=True)
plt.savefig('Fig.11(d).jpg', dpi=700, bbox_inches='tight',pad_inches=0)

shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],
                                       shap_value[3],
                                       feature_names = X_train.columns,
                                       max_display = 19
                                       )

shap.decision_plot(explainer.expected_value,
                   shap_value[3],
                   X_train.iloc[3]
                   )

# fraudent
shap.initjs()
shap.force_plot(explainer.expected_value[1],
                shap_value[15],
                X_train.iloc[15],
                text_rotation=20,
                matplotlib=True)

shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
                                       shap_value[15],
                                       feature_names = X_train.columns,
                                       max_display = 19)

shap.decision_plot(explainer.expected_value[1],
                   shap_value[15],
                   X_train.iloc[15])




