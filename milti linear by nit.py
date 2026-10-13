# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 11:11:40 2026

@author: dhuma

"""



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm

# Load dataset
dataset = pd.read_csv(r"D:\download\data.csv")

# Features and target
x = dataset.drop("Profit", axis=1)
y = dataset["Profit"]

# Encode categorical variable
x = pd.get_dummies(x, drop_first=True, dtype=int)

# Split dataset
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)


# Train model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predict
y_pred = regressor.predict(x_test)

# Model parameters
print("Intercept:", regressor.intercept_)
print("Coefficients:",regressor.coef_)
print(pd.Series(regressor.coef_, index=x.columns))

# Accuracy
print("R² Score:", r2_score(y_test, y_pred))

# OLS Summary (training data)
X_train = sm.add_constant(x_train)
ols_model = sm.OLS(y_train, X_train).fit()

import statsmodels.api as sm

# Convert DataFrame to NumPy
X = x.values

# Add constant only once
X = sm.add_constant(X)

# Step 1
X_opt = X[:, [0,1,2,3,4,5]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())

#Step 2 (remove variable with highest p-value)
X_opt = X[:, [0,1,2,3,5]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())

# Step 3
X_opt = X[:, [0,1,2,3]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())

# Step 4
X_opt = X[:, [0,1,2]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())

X_opt = X[:, [0,1]]
model = sm.OLS(y, X_opt).fit()
print(model.summary())


