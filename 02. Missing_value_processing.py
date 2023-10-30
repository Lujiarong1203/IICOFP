import numpy as np
import pandas as pd
import missingno as msno
import matplotlib.pyplot as plt



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data=pd.read_csv('data/data_1.csv')
print(data.head(5), '\n', data.shape)

# 特征Accepting_Currency_Num中，有153个样本为unknown，将其设为缺失值，以待下一步填充
index=data.Accepting_Currency_Num
print(index.value_counts())

data['Accepting_Currency_Num']=data['Accepting_Currency_Num'].apply(lambda x:
                                                                    np.NaN if x=='unknown'
                                                                    else int(x))

print(data.dtypes)
# 除了Country_Region外，其他特征均为浮点型和整数型
print(data.describe())

# 查看缺失值情况
na_ratio=data.isnull().sum()[data.isnull().sum()>=0].sort_values(ascending=False)/len(data)
na_sum=data.isnull().sum().sort_values(ascending=False)
print(na_ratio, '\n', na_sum)

missng1=msno.matrix(data,labels=True,label_rotation=0,fontsize=15,figsize=(15, 10))#绘制缺失值矩阵图
plt.xticks(fontsize=25, rotation=30)
plt.yticks(fontsize=25, rotation=30)
plt.savefig('Fig.4(a).jpg', bbox_inches='tight',pad_inches=0,dpi=1500,)
plt.show()

# 填充Accepting_Currency_Num的缺失值
M=data['Accepting_Currency_Num'].mode()

data['Accepting_Currency_Num']=data['Accepting_Currency_Num'].fillna(float(M))
na_sum_F=data.isnull().sum().sort_values(ascending=False)
print(na_sum_F)

data.to_csv(path_or_buf=r'data/data_2.csv', index=None)