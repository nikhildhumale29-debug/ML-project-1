# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 10:43:17 2026

@author: dhuma
"""

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso 
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\dhuma\Downloads\new cars.csv")
df.head()



# Replace '?' with NaN

df = pd.read_csv("new cars.csv")

df.replace('?', np.nan, inplace=True)

# Fill NaNs with median of each column
df = df.apply(lambda x: x.fillna(x.median()), axis=0)

print(df.head())

x=df.drop(['MPG'],axis=1) #indipendent veriable

y=df[['MPG']]  #dependent variable

x_s = preprocessing.scale(x)
x_s = pd.DataFrame(x_s,columns=x.columns)  # converting scaled data into frame

y_s = preprocessing .scale(y)
y_s = pd.DataFrame(y_s,columns=y.columns ) #ideally train test data shoud be in column

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.3, random_state=2)
x_train.shape

regression_model=LinearRegression()
regression_model.fit(x_train, y_train)

for idx,col_name in enumerate(x_train.columns):
    print('The coefficient for {} is {}'.format(col_name, regression_model.coef_[0][idx]))
    
intercept = regression_model.intercept_[0]
print('The intercept is {}'.format(intercept))    
     
ridge_model =Ridge(alpha = 0.3)
ridge_model.fit(x_train,y_train)        

print('Ridge model coef:{}'.format(ridge_model.coef_))

lasso_model =Lasso(alpha =0.1)
lasso_model.fit(x_train,y_train)

print('Lasso model coef:{}'.format(lasso_model.coef_))


# Predict
y_pred = regressor.predict(x_test)


# Accuracy
print("R² Score:", r2_score(y_test, y_pred))