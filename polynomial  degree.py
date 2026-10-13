# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 12:34:16 2026

@author: dhuma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
dataset =pd.read_csv(r"C:\Users\dhuma\Downloads\poly salary.csv")

x = dataset.iloc[:,1:2].values

y = dataset.iloc[:,2].values

#linearmodel --linear algor(degree-1)

from sklearn.linear_model import LinearRegression


lin_reg =LinearRegression()
lin_reg.fit(x,y)

#polynomial model(by defeault degree-2)

from sklearn.preprocessing import PolynomialFeatures


poly_reg =PolynomialFeatures(degree=2)
x_poly =poly_reg.fit_transform(x)

poly_reg.fit(x_poly, y) 

lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly, y)

#linear regression visualization
plt.scatter(x,y,color ='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title('Truth or bluff(LinearRegression)')
plt.xlabel('position level')
plt.ylabel('salary')      
plt.show()   

#poly nomial visualization
plt.scatter(x,y,color ='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.title('Truth or bluff(LinearRegression)')
plt.xlabel('position level')
plt.ylabel(' salary')      
plt.show()  

lin_model_pred =lin_reg.predict([[6.5]])
lin_model_pred

poly_model_pred =lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred


poly_reg =PolynomialFeatures(degree=3)
x_poly =poly_reg.fit_transform(x)

poly_reg.fit(x_poly, y) 

lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly, y)


#poly nomial visualization
plt.scatter(x,y,color ='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.title('Truth or bluff(LinearRegression)')
plt.xlabel('position level')
plt.ylabel(' salary')      
plt.show()  

lin_model_pred =lin_reg.predict([[6.5]])
lin_model_pred

poly_model_pred =lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred


poly_reg =PolynomialFeatures(degree=4)
x_poly =poly_reg.fit_transform(x)

poly_reg.fit(x_poly, y) 

lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly, y)


#poly nomial visualization
plt.scatter(x,y,color ='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.title('Truth or bluff(LinearRegression)')
plt.xlabel('position level')
plt.ylabel(' salary')      
plt.show()  

lin_model_pred =lin_reg.predict([[6.5]])
lin_model_pred

poly_model_pred =lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred


poly_reg =PolynomialFeatures(degree=5)
x_poly =poly_reg.fit_transform(x)

poly_reg.fit(x_poly, y) 

lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly, y)


#poly nomial visualization
plt.scatter(x,y,color ='red')
plt.plot(x, lin_reg_2.predict(poly_reg.fit_transform(x)))
plt.title('Truth or bluff(LinearRegression)')
plt.xlabel('position level')
plt.ylabel(' salary')      
plt.show()  

lin_model_pred =lin_reg.predict([[6.5]])
lin_model_pred

poly_model_pred =lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))
poly_model_pred


