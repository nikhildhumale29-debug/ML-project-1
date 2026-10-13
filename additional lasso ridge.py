# -*- coding: utf-8 -*-
"""
Created on Fri Jun 12 18:45:01 2026

@author: dhuma
"""


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso 

# Load dataset
df = pd.read_csv(r"C:\Users\dhuma\Downloads\new cars.csv")

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert all columns to numeric where possible
df = df.apply(pd.to_numeric, errors='ignore')

# Fill NaNs with median of each column
df = df.apply(lambda x: x.fillna(x.median()), axis=0)

print(df.head())

# Split into features and target
x = df.drop(['MPG'], axis=1)   # independent variables
y = df[['MPG']]                # dependent variable

# Scale features (optional)
x_s = preprocessing.scale(x)
x_s = pd.DataFrame(x_s, columns=x.columns)

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=2)

# Linear Regression
regression_model = LinearRegression()
regression_model.fit(x_train, y_train)

for idx, col_name in enumerate(x_train.columns):
    print(f"The coefficient for {col_name} is {regression_model.coef_[0][idx]}")

print(f"The intercept is {regression_model.intercept_[0]}")

# Ridge Regression
ridge_model = Ridge(alpha=0.3)
ridge_model.fit(x_train, y_train)
print("Ridge model coefficients:", ridge_model.coef_)

# Lasso Regression
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train)
print("Lasso model coefficients:", lasso_model.coef_)

from sklearn.metrics import r2_score

# Predictions
y_pred_linear = regression_model.predict(x_test)
y_pred_ridge  = ridge_model.predict(x_test)
y_pred_lasso  = lasso_model.predict(x_test)

# R² scores
print("Linear Regression R²:", r2_score(y_test, y_pred_linear))
print("Ridge Regression R²:", r2_score(y_test, y_pred_ridge))
print("Lasso Regression R²:", r2_score(y_test, y_pred_lasso))

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Predictions
y_pred_linear = regression_model.predict(x_test)
y_pred_ridge  = ridge_model.predict(x_test)
y_pred_lasso  = lasso_model.predict(x_test)

# R² scores
r2_linear = r2_score(y_test, y_pred_linear)
r2_ridge  = r2_score(y_test, y_pred_ridge)
r2_lasso  = r2_score(y_test, y_pred_lasso)

print("Linear Regression R²:", r2_linear)
print("Ridge Regression R²:", r2_ridge)
print("Lasso Regression R²:", r2_lasso)

# Plot comparison
models = ['Linear', 'Ridge', 'Lasso']
scores = [r2_linear, r2_ridge, r2_lasso]

plt.bar(models, scores, color=['blue', 'green', 'red'])
plt.ylabel("R² Score")
plt.title("Comparison of Regression Models")
plt.ylim(0, 1)  # keep scale between 0–1
plt.show()

