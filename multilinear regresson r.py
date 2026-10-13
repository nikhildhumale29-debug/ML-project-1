# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:54:47 2026

@author: dhuma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load dataset
dataset = pd.read_csv(r"D:\download\data.csv")

# Features (all columns except Profit)
x = dataset.iloc[:, :-1]

# Target (Profit column)
y = dataset.iloc[:, 4]

# Convert categorical 'State' into dummy variables
x = pd.get_dummies(x, dtype=int)

# Split data BEFORE fitting
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

# Create and fit model
regressor = LinearRegression()
regressor.fit(x_train, y_train)
                             
# Predict
y_pred = regressor.predict(x_test)

# Coefficients and intercept
m = regressor.coef_
c = regressor.intercept_

print("Coefficients:", m)
print("Intercept:", c)


import statsmodels.api as sm

# Convert DataFrame to NumPy arrays
X= x.values
Y= y.values

# Add constant column for intercept
X_opt = sm.add_constant(X)

# Fit OLS model
regressor_OLS = sm.OLS(Y, X_opt).fit()

# Summary
print(regressor_OLS.summary())

from sklearn.metrics import r2_score

print("R² Score:", r2_score(y_test, y_pred))



 




