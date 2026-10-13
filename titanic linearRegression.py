# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 15:22:00 2026

@author: dhuma
"""

import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ----------------------------
# Load Titanic dataset
# ----------------------------
df = sns.load_dataset("titanic")

# Quick look
print(df.head())

# ----------------------------
# Preprocessing
# ----------------------------
# Drop rows with missing target (fare)
df = df.dropna(subset=["fare"])

# Select features (example: age, sex, class, sibsp, parch)
X = df[["age", "sex", "class", "sibsp", "parch"]]
y = df["fare"]

# Handle missing values in age
X["age"] = X["age"].fillna(X["age"].median())

# Convert categorical variables to numeric (one-hot encoding)
X = pd.get_dummies(X, drop_first=True)

# ----------------------------
# Train-Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split( X, y,test_size=0.2, random_state=42)

# ----------------------------
# Fit Linear Regression
# ----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# ----------------------------
# Evaluate
# ----------------------------
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
