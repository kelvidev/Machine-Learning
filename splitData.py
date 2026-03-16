import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('data/iris.csv')

X = dataset.drop(columns=["target"])
y = dataset["target"].values

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train, X_test, Y_train, Y_test)