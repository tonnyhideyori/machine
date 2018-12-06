#import library
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

names = ["Bob", "Edwin", "Fides", "Beatus"]
birth = [124, 45, 11, 65]
BabyDataSet = list(zip(names, birth))
print(BabyDataSet)

df = pd.DataFrame(data=BabyDataSet, columns=["Names", "Births"])
df.to_csv("births1900", index=False, header=False)
df = pd.read_csv("births1900", names=["Names", "Births"])
Sorted = df.sort_values(["Births"], ascending=True)
Sorted.head(1)
s1 = pd.Series(np.random.randn(5), index=list(range(0, 10, 2)))
s1
y = s1.iloc[1:]
df1 = pd.DataFrame(np.random.randn(6, 4),
                   index=list(range(0, 12, 2)),
                   columns=list(range(0, 8, 2)))
y=df1.iloc[:2,:4]
y