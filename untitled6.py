# -*- coding: utf-8 -*-
"""
Created on Fri May 22 12:38:23 2026

@author: dhuma
"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# loard dataset
df = pd.read_csv(r"C:\Users\dhuma\Downloads\placement.csv")

#quik look
print(df.head()) 

#FEACHERS AND TARGET
x= df.iloc[:,0:1]
y= df.iloc[:,-1] 

# Scatter plot

plt.scatter(df['cgpa'], df['package'])
plt.xlabel('cgpa')
plt.ylabel('package(in lpa)')
plt.title("Student Anaylysis")

#train test
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Linear regression
lr = LinearRegression()
lr.fit(x_train, y_train)

# Scatter plot of actual data
plt.scatter(x, y, color="blue", label="Actual data")

# Regression line (fitted on all x values)
plt.plot(x, lr.predict(x), color="red", linewidth=2, label="Regression line")


class MeraLR:
    
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self,X_train,y_train):
        X_train = np.insert(X_train,0,1,axis=1)
        
        # calcuate the coeffs
        betas = np.linalg.inv(np.dot(X_train.T,X_train)).dot(X_train.T).dot(y_train)
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
    
    def predict(self,X_test):
        y_pred = np.dot(X_test,self.coef_) + self.intercept_
        return y_pred
        lr = MeraLR()
        lr.fit(X_train,y_train)
        X_train.shape
        np.insert(X_train,0,1,axis=1).shape
        y_pred = lr.predict(X_test)
        r2_score(y_test,y_pred)
        lr.coef_
        lr.intercept_
        
    