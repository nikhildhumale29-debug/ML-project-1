# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 18:58:40 2026

@author: dhuma
"""


import xgboost
%pip install xgboost

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load dataset
dataset = pd.read_csv(r"C:\Users\dhuma\Downloads\employees_400.csv")

# Features (Age, Estimated Salary) and Target (Purchased)
x = dataset.loc[:, ['Age', 'Income']].values
y = dataset.iloc[:, -1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=50)

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

#random = RandomForestClassifier(n_estimators=100, max_depth=6)
#random.fit(x_train, y_train)
#y_pred = random.predict(x_test)

xgb = XGBClassifier(n_estimators=100, max_depth=6)
xgb.fit(x_train, y_train)
y_pred = xgb.predict(x_test)

#classifier = DecisionTreeClassifier()
#classifier.fit(x_train, y_train)

# Predicting the test set results
#y_pred = classifier.predict(x_test) 

# Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)

# Accuracy
ac = accuracy_score(y_test, y_pred)
print("Accuracy:", ac)

# Classification Report
cr = classification_report(y_test, y_pred)
print("Classification Report:\n", cr)

# Bias (train score) and Variance (test score)
bias = xgb.score(x_train, y_train)
variance = xgb.score(x_test, y_test)
print("Bias (Train Score):", bias)
print("Variance (Test Score):", variance)


%pip install lazypredict

from lazypredict.Supervised import LazyClassifier

# Create LazyClassifier
clf = LazyClassifier(verbose=0, ignore_warnings=True)

# Train and evaluate all classifiers
models, predictions = clf.fit(
    x_train,
    x_test,
    y_train,
    y_test
)

print(models)
