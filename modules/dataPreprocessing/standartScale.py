import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import utils.analyzeData as analyzer

dataframe = pd.read_csv("data/winequality-red.csv", sep=";")

X = dataframe.drop(columns=["quality"], )
y = dataframe["quality"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42, )

std = StandardScaler()

analyzer.findCategoricalData(dataset=dataframe)

X_train = std.fit_transform(X_train)
X_test = std.transform(X_test)

print(X_train)
print(X_test)