from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("./Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
"""x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)"""
# making linear regression
linear = LinearRegression()
linear.fit(x, y)
# polynomial linear regression
poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(x)
linear_reg = LinearRegression()
linear_reg.fit(x_poly, y)
plt.scatter(x, y, color="red")
plt.plot(x, linear.predict(x), color="blue")
plt.title("Truth or bluff[linear regression]")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, linear_reg.predict(poly_reg.fit_transform(x_grid)), color="blue")
plt.title("Truth or bluff[polynomial regression]")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()
# predict the salary
# linear.predict(6.5)
# predicting by polynomial
linear_reg.predict(poly_reg.fit_transform(6.5))
