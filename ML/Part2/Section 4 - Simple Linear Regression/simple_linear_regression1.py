# Data Preprocessing Template

# Importing the libraries
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
#predicted value
y_pred=regressor.predict(X_test)
#ploting 
plt.scatter(X_train,y_train,color="red")
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title('salary VS experience[training set]')
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()
#ploting for prediction
plt.scatter(X_test,y_test,color="red")
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title('salary VS experience[test set]')
plt.xlabel("years of experience")
plt.ylabel("Salary")
plt.show()
