# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 19:44:36 2026

@author: dhuma
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Define the model
rf = RandomForestClassifier(random_state=42)

# Define parameter grid to search
param_grid = {
    'n_estimators': [50, 100, 200],       # Number of trees
    'max_depth': [None, 5, 10],           # Tree depth
    'min_samples_split': [2, 5, 10]       # Min samples to split a node
}

# Create GridSearchCV object
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,                # 5-fold cross-validation
    scoring='accuracy',  # Metric to optimize
    n_jobs=-1,           # Use all CPU cores
    verbose=1            # Show progress
)

# Fit the model
grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# Evaluate on test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Test Accuracy:", accuracy_score(y_test, y_pred))
