# -*- coding: utf-8 -*-
"""
Linear Regression
Created on Thu Nov  4 06:50:36 2021
@author: Bagheri
"""
# Some Keys for Spyder:
# F9: run current line & selected lines
# Ctrl + I: help
# Ctrl + /: show more alternative options
# Ctrl + 1: for comment

import numpy as np
#import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

data = np.genfromtxt('datasets/house_price.txt', delimiter=',')
data.shape
data
# print(data)

x = data[:, 0]  
y = data[:, 1]

plt.figure(figsize=(10, 7))
plt.scatter(x, y, s=50, c='b', marker='*')
plt.title('House Dataset')
plt.xlabel('size')
plt.ylabel('price')
plt.show()

x = x.reshape(-1,1)  #change rows to columns
y = y.reshape(-1,1)
reg = LinearRegression()
reg.fit(x,y)

y_pred = reg.predict(x)
plt.scatter(x,y)
plt.plot(x,y_pred)
plt.show()
