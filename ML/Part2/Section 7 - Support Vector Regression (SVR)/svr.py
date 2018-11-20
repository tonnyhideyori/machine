

from sklearn.preprocessing import  StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVR


dataset = pd.read_csv("./Position_Salaries.csv")
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2:3].values
sc_x=StandardScaler()
sc_y=StandardScaler()

x=sc_x.fit_transform(x)
y=sc_y.fit_transform(y)


reg = SVR(kernel="rbf")
reg.fit(x, y)
x_grid = np.arange(min(x), max(x), 0.01)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color="red")
plt.plot(x_grid, reg.predict(x_grid), color="blue")
plt.title("Truth or bluff[SVR]")
plt.xlabel("position level")
plt.ylabel("Salary")
plt.show()
y_pred = sc_y.inverse_transform(reg.predict(sc_x.transform(np.array([[6.5]]))))
