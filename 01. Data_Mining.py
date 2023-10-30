import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data=pd.read_csv('data/Data Set.csv')
print(data.head(10), '\n', data.shape)
print(data.dtypes)

index=data.goal
print(index.value_counts())

index=data.country_region
print(index.value_counts())

index=data.platform
print(index.value_counts())

index=data.acceptingCurrencyNum
print(index.value_counts())

index=data.whitepaper
print(index.value_counts())

index=data.video
print(index.value_counts())

index=data.socialMedia
print(index.value_counts())

index=data.GitHub
print(index.value_counts())

index=data.CEOPhoto
print(index.value_counts())

"""
处理标签变量，将N设为0，Y设为1
"""
label_mapping = {"N": 0, "Y": 1}

data['Goal']=data['goal'].apply(lambda x:
                                label_mapping.get(x))
print(data['Goal'].head(10))
"""
   处理时间，计算出发售的天数
"""
import datetime
data['End_Date']=data['enddate'].apply(lambda x:
                                       datetime.datetime.strptime(x, "%d-%m-%Y"))

data['Start_Date']=data['startdate'].apply(lambda y:
                                           datetime.datetime.strptime(y, "%d-%m-%Y"))

print(data[['End_Date', 'Start_Date']].head(10))

# 找出最早和最晚的日期
earliest_date = min(data['End_Date'])
latest_date = max(data['End_Date'])

print("最早的日期:", earliest_date.strftime("%d-%m-%Y"))
print("最晚的日期:", latest_date.strftime("%d-%m-%Y"))

data['Offering_days']=data['End_Date']-data['Start_Date']

data['Offering_days']=data['Offering_days'].apply(lambda x:
                                                  x.total_seconds()/60/60/24)
print(data['Offering_days'].head(10))



# 发售天数中有负值，可能是数据记录错误，将其加绝对值变为正的
index=data.Offering_days
print(index.value_counts())

data['Offering_days']=data['Offering_days'].apply(lambda x:
                                                  np.abs(x) if x<0
                                                  else x)

"""
处理CoinNum，将科学计数法转换为普通数值
"""
data['Coin_Num']=data['coinNum'].apply(lambda x:
                                       "{:.0f}".format(x))
print(data['Coin_Num'].head(10))

"""
处理country_region，因为其中含有同一个国家的不同名称，将其统一
"""
data['country_region']=data['country_region'].apply(lambda x:
                                                    'USA' if x=='United States' or x=='United States of America'
                                                    else x)
data['country_region']=data['country_region'].apply(lambda x:
                                                    'UK' if x=='United Kingdom'
                                                    else x)
data['country_region']=data['country_region'].apply(lambda x:
                                                    'UAE' if x=='United Arab Emirates'
                                                    else x)
data['country_region']=data['country_region'].apply(lambda x:
                                                    'Russia' if x=='Russian'
                                                    else x)
data['country_region']=data['country_region'].apply(lambda x:
                                                    'Peru' if x=='Perú'
                                                    else x)
data['country_region']=data['country_region'].apply(lambda x:
                                                    'Netherland' if x=='Netherlands'
                                                    else x)
data['country_region']=data['country_region'].apply(lambda x:
                                                    'Cayman Island' if x=='Cayman Islands'
                                                    else x)
index=data.country_region
print(index.value_counts())

input()


"""
处理categories，统计出涉及种类的个数
"""
data['Category_Num']=data['categories'].apply(lambda x:
                                              len(x.split(',')) if x is not np.NaN
                                              else x)
print(data['Category_Num'])

"""
处理teamLinkedIn 和 teamPhotos，将百分数转换为小数
"""
data['Team_LinkedIn']=data['teamLinkedIn'].apply(lambda x:
                                                 float(x.strip('%')) / 100)
data['Team_Photos']=data['teamPhotos'].apply(lambda x:
                                             float(x.strip('%')) / 100)
# print(data[['Team_LinkedIn', 'Team_Photos']])

"""
特征Platform中，Ethereum有979个，因此，选择将其设为1，其他为0
"""
# index=data.Platform
# print(index.value_counts())

Platform_keywords = ['Ethereum', 'ETH']
data['platform']=data['platform'].str.contains('|'.join(Platform_keywords), na=False).astype(float)



data_1=data[['Goal', 'Offering_days', 'Coin_Num', 'teamSize', 'country_region',
             'Category_Num', 'overallrating', 'ratingTeam', 'ratingProduct', 'platform',
             'acceptingCurrencyNum', 'whitepaper', 'video', 'socialMedia', 'GitHub',
             'Team_LinkedIn', 'Team_Photos', 'CEOPhoto']]

data_1.rename(columns={'teamSize': 'Team_Size', 'country_region': 'Country_Region',
                       'overallrating': 'Overall_Rating', 'ratingProduct': 'Product_Rating',
                       'ratingTeam': 'Team_Rating', 'platform': 'Platform',
                       'acceptingCurrencyNum': 'Accepting_Currency_Num', 'whitepaper': 'Whitepaper',
                       'video': 'Video', 'socialMedia': 'Social_Media',
                       'CEOPhoto': 'CEO_Photo'}, inplace=True)


print(data_1.head(10), '\n', data_1.shape)

print(data_1.Coin_Num)

data_1.to_csv(path_or_buf=r'data/data_1.csv', index=None)