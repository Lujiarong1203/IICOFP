import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

data=pd.read_csv('data/data_3.csv')
print(data.head(5), '\n', data.shape)

"""
绘制几个数值型特征的分布图
"""
num_col_1= ['Team_Size', 'Overall_Rating', 'Accepting_Currency_Num', 'Team_Photos']
print(num_col_1)

dist_cols = 2
dist_rows = len(num_col_1)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in num_col_1:
    ax = plt.subplot(2, 2, i)
    ax = sns.kdeplot(data=data[data.Goal==0][col], bw=1, label="Failed", color="darkblue", shade=True)
    ax = sns.kdeplot(data=data[data.Goal==1][col], bw=0.5, label="Successful", color="darkorange", shade=True)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    i += 1
plt.show()



"""
绘制0-1型特征的分布图
"""
str_col_1= ['Platform', 'Whitepaper', 'Social_Media', 'GitHub']
print(str_col_1)

dist_cols = 2
dist_rows = len(str_col_1)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in str_col_1:
    ax=plt.subplot(3, 2, i)
    ax=sns.countplot(x=data[col], hue="Goal", data=data, palette=['cornflowerblue', 'darkorange'])
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    ax.get_legend().remove()
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    # plt.legend(loc='upper left')

    i += 1
    plt.tight_layout();
plt.show()



str_col_2= ['GitHub', 'CEO_Photo']
print(str_col_2)

dist_cols = 2
dist_rows = len(str_col_2)
plt.figure(figsize=(4 * dist_cols, 4 * dist_rows))

i = 1
plt.rcParams.update({'font.size': 15})
for col in str_col_2:
    ax=plt.subplot(1, 2, i)
    ax=sns.countplot(x=data[col], hue="Goal", data=data)
    ax.set_xlabel(col, fontdict={'weight': 'normal', 'size': 15})
    ax.set_ylabel("Frequency", fontdict={'weight': 'normal', 'size': 15})
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    plt.legend(loc='upper left')

    i += 1
    plt.tight_layout();
plt.show()