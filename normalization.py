#7107029058_(1002 hw)
#描述性統計 2化散佈圖 => 針對想預測的欄位 3相關係數 4資料正規化(z-score) 5資料是否有遺漏值 6類別值 7線性回歸 8.MSE

from sklearn import datasets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression


#download dataset
avocado=pd.read_csv("avocado.csv")
print(avocado.shape)
print(avocado.keys())

#describe data
print(avocado.head())
print(avocado.describe())
print(avocado.info())

#Scatter
plt.scatter(avocado['Total Volume'],avocado['Total Bags'],color='blue')
plt.show()


#相關係數
x=avocado['Total Volume']
y=avocado['Total Bags']

n = len(avocado['Total Volume'])
x_mean = avocado['Total Volume'].mean()
y_mean = avocado['Total Bags'].mean()

diff = (x-x_mean)*(y-y_mean)
covar = diff.sum()/n
print("共變異數:", covar)

corr = covar/(x.std()*y.std())
print("相關係數:", corr)

df = pd.DataFrame({"Total Volume":x,
                   "Total Bags":y})
print(df.corr())



#資料標準化

#檢查有無遺漏值（都為0）
print(avocado.isnull().sum())
print(sum(avocado["AveragePrice"].isnull()))

#分類值
df = pd.read_csv("avocado.csv")
print(df)
size_mapping = {"XXL": 5,
                "XL": 4,
                "L": 3,
                "M": 2,
                "S": 1,
                "XS": 0}

df["Total Volume"] = df["Total Volume"].map(size_mapping)
print(df)


label_encoder = preprocessing.LabelEncoder()
df["Total Bags"] = label_encoder.fit_transform(df["Total Bags"])
print(df)


#線性迴歸
x = avocado['Total Volume'].values
y = avocado['Total Bags'].values

slr = LinearRegression()
slr.fit(x, y)

#y_pred = slr.predict(x)
print('Slope: %.3f' % slr.coef_[0])
print('Intercept: %.3f' % slr.intercept_)

def lin_regplot(X, y, model):
    plt.scatter(X, y, c='lightblue')
    plt.plot(X, model.predict(X), color='red', linewidth=2)    
    return 

lin_regplot(x, y, slr)
avocado.reshape(-1, 1)
plt.xlabel('Total Volume')
plt.ylabel('Total Bags')
#plt.tight_layout()
# plt.savefig('./figures/scikit_lr_fit.png', dpi=300)
plt.show()


#MSE
