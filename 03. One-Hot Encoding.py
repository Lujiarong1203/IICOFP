import numpy as np
import pandas as pd


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


data=pd.read_csv('data/data_2.csv')
print(data.shape, '\n', data.isnull().sum())

index=data.Country_Region
print(index.value_counts())

Country_Num=set(data['Country_Region'])
print('出现的国家数：', len(Country_Num))

dummy = pd.get_dummies(data['Country_Region'])
data_3= pd.concat([data, dummy], axis=1)

# Generate the latest dataset and view the data types for each feature
data_3.drop('Country_Region', axis=1, inplace=True)

data_3.columns=data_3.columns.str.replace('[^\w\s]', '_')
data_3.columns=data_3.columns.str.replace(' ', '')
print(data_3.dtypes, '\n', data_3.head(10), '\n', data_3.shape)

data_3.to_csv(path_or_buf=r'data/data_3.csv', index=None)