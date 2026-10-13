# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 19:55:16 2026

@author: dhuma
"""

from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
# Load data
data = fetch_california_housing()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target
# Split and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train, y_train)
# Evaluate
y_pred = reg.predict(X_test)
print("MSE:", mean_squared_error(y_test, y_pred))
print("R²:", r2_score(y_test, y_pred))