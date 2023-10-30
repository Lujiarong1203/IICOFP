import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


data=pd.read_csv('data/data_3.csv')
print(data.head(5), '\n', data.shape)

# plt.boxplot(x = data.Coin_Num, # 指定绘制箱线图的数据
#          whis = 1.5, # 指定1.5倍的四分位差
#          widths = 0.7, # 指定箱线图的宽度为0.8
#          patch_artist = True, # 指定需要填充箱体颜色
#          showmeans = True, # 指定需要显示均值
#          boxprops = {'facecolor':'steelblue'}, # 指定箱体的填充色为铁蓝色
#         # 指定异常点的填充色、边框色和大小
#          flierprops = {'markerfacecolor':'red', 'markeredgecolor':'red', 'markersize':4},
#          # 指定均值点的标记符号（菱形）、填充色和大小
#          meanprops = {'marker':'D','markerfacecolor':'black', 'markersize':4},
#          medianprops = {'linestyle':'--','color':'orange'}, # 指定中位数的标记符号（虚线）和颜色
#          labels = [''] # 去除箱线图的x轴刻度值
#          )
# # 显示图形
# plt.show()
#
index=data.Coin_Num
print(index.value_counts())
#
# print('从大到小排序：', data['Coin_Num'].sort_values())
#
#
# Q1 = data.Coin_Num.quantile(q = 0.25)
# Q3 = data.Coin_Num.quantile(q = 0.75)
# print('第1、3分位点：', Q1, Q3)
#
# low_whisker = Q1 - 3*(Q3 - Q1)
# up_whisker = Q3 + 3*(Q3 - Q1)
# print('下须，上须：', low_whisker, up_whisker)
#
# print('1万-1000亿之外的')
# print(data.Coin_Num[(data.Coin_Num > 100000000000) | (data.Coin_Num < 10000)])
#
#
# print('3_Sigma')
# mean = data['Coin_Num'].mean()
# std = data['Coin_Num'].std()
# print('均值、标准差：', mean, std)
#
# # 定义异常值的阈值（例如，将超过3个标准差之外的值定义为异常值）
# threshold = 3 * std
# print('3_Sigma', threshold)
#
# # 检测异常值的索引
# outlier_indices = data[(data['Coin_Num'] - mean).abs() > threshold].index
#
# # 打印异常值的索引
# print("异常值索引：", outlier_indices)




# 寻找异常点
outliers=(data['Coin_Num'] < 1000).sum()
print('异常值数量：', outliers)

# 将异常的值设为缺失值，以待后续填充
data['Coin_Num']=data['Coin_Num'].apply(lambda x:
                                        np.NAN if x < 1000
                                        else x)

print('设为缺失值后的数量：', data['Coin_Num'].isnull().sum())

Group=data.groupby('Goal')

fill_Coun_Num = Group['Coin_Num'].apply(lambda x: x.fillna(x.mean()))

data['Coin_Num']=pd.DataFrame(fill_Coun_Num)
print(data.isnull().sum())

# outliers=(data['Coin_Num'] < 5000).sum()
# print('填充后的异常值数量：', outliers)

"""
处理CoinNum，将科学计数法转换为普通数值
"""
# data['Coin_Num']=data['Coin_Num'].apply(lambda x:
#                                        "{:.0f}".format(x))

print('填充后：')
# print(len(data.Coin_Num[(data.Coin_Num > 100000000000) | (data.Coin_Num < 10000)]))


print(data['Coin_Num'].head(10))
data.to_csv(path_or_buf=r'data/data_4.csv', index=None)