import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

plt.rc('font',family='Times New Roman')
plt.rcParams['font.sans-serif']=['SimHei']

"""
绘制Country_Region的词频图
"""
data=pd.read_csv('data/data_1.csv')
print(data.head(10), '\n', data.shape)

data.sort_values(by=['Goal'], inplace=True)
print(data.head(10), '\n', data.tail(10))

index=data.Goal
print(index.value_counts())

# 将特征转换为列表
Country_Region_Text=list(data['Country_Region'])
data_Faild=Country_Region_Text[:605]
data_Successful=Country_Region_Text[605:]
print(data_Faild, '\n', data_Successful)

# 失败的云图
wc=WordCloud(max_words=1000, width=1600, height=1000, collocations=False).generate(" ".join(data_Faild))
plt.figure(figsize=(20, 20))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.imshow(wc)
plt.show()

# 成功的云图
wc=WordCloud(max_words=1000, width=1600, height=1000, collocations=False).generate(" ".join(data_Successful))
plt.figure(figsize=(20, 20))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.imshow(wc)
plt.show()


"""
绘制Category_Num的词云图
"""
data=pd.read_csv('data/Data Set.csv')
print(data.head(10), data.shape)

data.sort_values(by=['goal'], inplace=True)
print(data.head(10), '\n', data.tail(10))
#
index=data.goal
print(index.value_counts())
#
# 将特征转换为列表
Category_Num_Text=list(data['categories'])
data_Faild=Category_Num_Text[:605]
data_Successful=Category_Num_Text[605:]
print(data_Faild, '\n', data_Successful)
#
# # 失败的云图
wc=WordCloud(max_words=1000, width=1600, height=1000, collocations=False).generate(" ".join(data_Faild))
plt.figure(figsize=(20, 20))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.imshow(wc)
plt.show()
#
# # 成功的云图
wc=WordCloud(max_words=1000, width=1600, height=1000, collocations=False).generate(" ".join(data_Successful))
plt.figure(figsize=(20, 20))
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.imshow(wc)
plt.show()