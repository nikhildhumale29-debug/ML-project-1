# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 13:20:11 2026

@author: dhuma
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer


df = pd.read_csv(r"C:\Users\dhuma\Downloads\dataset.csv")

print(df.head())

x = df.iloc[:,:-1].values

y = df.iloc[:,3].values


imputer = SimpleImputer()

imputer = imputer.fit(x[:,1:3])

x[:,1:3] =imputer.transform(x[:,1:3])

from sklearn.preprocessing import LabelEncoder

labelencoder_x = LabelEncoder()

labelencoder_x.fit_transform(x[:,0])
x[:,0] =labelencoder_x.fit_transform(x[:,0]) 


labelencoder_y = LabelEncoder()
y =labelencoder_y.fit_transform(y) 

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2, random_state=0)
