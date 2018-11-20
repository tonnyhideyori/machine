# Data preprocessing

# importing libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer, StandardScaler, LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("../Data.csv")
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0)

"""
-----this is for scaling---
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transform(x_test)"""

"""for template we remove these line of correcting missing dataset
#imputer = Imputer(missing_values="NaN", strategy="mean", axis=0)
#imputer = imputer.fit(x[:, 1:3])
#x[:, 1:3] = imputer.transform(x[:, 1:3])
#print(x)
#lebalencoder_x=LabelEncoder()
#x[:,0]=lebalencoder_x.fit_transform(x[:,0])
#onehotencoder=OneHotEncoder(categorical_features=[0])
#x=onehotencoder.fit_transform(x).toarray()
#labelencoder_y=LabelEncoder()
#y=labelencoder_y.fit_transform(y)
#---------------end of correcting and encoding------#
"""
