# -*- coding: utf-8 -*-
"""
Created on Sun Jul 26 14:24:44 2026

@author: dhuma
"""
# ------------------------------
# ---------------------------------------------
# Decision Tree Analysis: Carseats Dataset (One-Hot Encoding)---------------------------------------------------

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# 1. Load data
df = pd.read_csv(r"D:\download\Company_Data.csv")

# 2. Target engineering
df["SalesCat"] = np.where(df["Sales"] > 8, "High", "Low")

# 3. One-hot encode categorical predictors
df_enc = pd.get_dummies(df, columns=["ShelveLoc", "Urban", "US"], drop_first=True)

# Features and target
X = df_enc.drop(columns=["Sales", "SalesCat"])
y = df_enc["SalesCat"].map({"High": 1, "Low": 0})

# 4. Train/test split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=42, stratify=y)

# 5. Hyperparameter tuning
param_grid = {
    "max_depth": [3, 4, 5, 6, None],
    "min_samples_leaf": [1, 5, 10],
    "criterion": ["gini", "entropy"],
}

grid = GridSearchCV ( DecisionTreeClassifier(random_state=42),param_grid,cv=5,scoring="accuracy")
grid.fit(X_train, y_train)

best_model = grid.best_estimator_
print("Best hyperparameters:", grid.best_params_)
print("Best CV accuracy:", round(grid.best_score_, 3))

# 6. Evaluate
y_pred = best_model.predict(X_test)
print("\nTrain accuracy:", round(accuracy_score(y_train, best_model.predict(X_train)), 3))
print("Test accuracy :", round(accuracy_score(y_test, y_pred), 3))
print("\nClassification report:\n", classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

# 7. Feature importance
importance = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)
print("\nFeature importances:\n", importance)

# 8. Visualizations
fig, ax = plt.subplots(figsize=(26, 14))
plot_tree(best_model,
          feature_names=X.columns,
          class_names=["Low", "High"],
          filled=True,
          rounded=True,
          fontsize=8,
          max_depth=3)
plt.title("Decision Tree (One-Hot Encoding) — top 3 levels")
plt.tight_layout()
plt.savefig("decision_tree_alt.png", dpi=150)
plt.close()

fig, ax = plt.subplots(figsize=(8, 6))
importance.sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
ax.set_title("Feature Importance — One-Hot Encoded Predictors")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("feature_importance_alt.png", dpi=150)
plt.close()

print("\nSaved: decision_tree_alt.png, feature_importance_alt.png")
