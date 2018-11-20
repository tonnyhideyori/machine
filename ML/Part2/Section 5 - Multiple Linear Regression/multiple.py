# Multiple linear regression
# importing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("./50_Startups.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

labelencoder_x = LabelEncoder()
x[:, 3] = labelencoder_x.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()
#avoiding the dummy variable trap
x=x[:,1:]
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=1/5, random_state=0)
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)

#building the optimal model using backward elimination
import statsmodels.formula.api as sn
x = np.append(arr=np.ones((50, 1)).astype(int), values=x, axis=1)
x_opt = x[:,[0,1,2,3,4,5]]
regressor_OLS=sn.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,1,3,4,5]]
regressor_OLS=sn.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3,4,5]]
regressor_OLS=sn.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3,5]]
regressor_OLS=sn.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()
x_opt = x[:,[0,3]]
regressor_OLS=sn.OLS(endog=y,exog=x_opt).fit()
regressor_OLS.summary()