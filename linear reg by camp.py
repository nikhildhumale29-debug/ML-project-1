# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 16:25:21 2026

@author: dhuma
"""

import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv(r"C:\Users\dhuma\Downloads\placement.csv")
print(df.head())

# Features and target
X = df.iloc[:,0:1].values   # cgpa
y = df.iloc[:,-1].values    # package

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)

# --- Custom Linear Regression ---
class MeraLR:
    def __init__(self):
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, X_train, y_train):
        # Add bias column
        X_train_bias = np.insert(X_train, 0, 1, axis=1)
        betas = np.linalg.inv(X_train_bias.T @ X_train_bias) @ X_train_bias.T @ y_train
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]
    
    def predict(self, X_test):
        return np.dot(X_test, self.coef_) + self.intercept_

# Train custom model
lr_custom = MeraLR()
lr_custom.fit(X_train, y_train)

# Predictions
y_pred_custom = lr_custom.predict(X_test)

# R² score
print("Custom MeraLR R²:", r2_score(y_test, y_pred_custom))

# --- Graph ---
plt.scatter(X_test, y_test, color="blue", label="Actual")
plt.plot(X_test, y_pred_custom, color="red", linewidth=2, label="Predicted")
plt.xlabel("CGPA")
plt.ylabel("Package (in LPA)")
plt.title("Custom Linear Regression (MeraLR)")
plt.legend()
plt.show()
