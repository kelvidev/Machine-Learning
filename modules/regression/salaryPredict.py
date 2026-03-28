import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

dataframe = pd.read_csv("data/years-salary.csv")

X = dataframe.drop(columns=["Salary"])
y = dataframe["Salary"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

# scaler = MinMaxScaler()

# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)

regressor = LinearRegression()

regressor.fit(X=X_train, y=y_train)

new_prediction = regressor.predict([[5.5]])

y_prediction = regressor.predict(X_test)

print(f"The accuracy was {(r2_score(y_pred= y_prediction, y_true=y_test) * 100):.2f}%" )


plt.scatter(X_train, y_train)
plt.scatter(X_test, y_test, color='green')
plt.plot(X_train, regressor.predict(X_train), color='green')
plt.show()

