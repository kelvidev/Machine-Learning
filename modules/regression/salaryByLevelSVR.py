import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
def main():

    df = pd.read_csv('data/Position_Salaries.csv')
    
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    
    df['Level'] = feature_scaler.fit_transform(df['Level'].to_numpy().reshape(-1,1))
    df['Salary'] = target_scaler.fit_transform(df['Salary'].to_numpy().reshape(-1,1))

    model = SVR(kernel='rbf', C=1, epsilon=0.1, )
    model.fit(df['Level'].to_numpy().reshape(-1,1), df['Salary'].to_numpy())
    
    print(target_scaler.inverse_transform([model.predict(feature_scaler.transform([[6.5]]))]))
    X = feature_scaler.inverse_transform([df['Level']]).reshape(-1)
    y_predicted = target_scaler.inverse_transform([model.predict(df['Level'].to_numpy().reshape(-1,1))]).reshape(-1)
    print(X)
    print(y_predicted)
    plt.plot(X, y_predicted, )
    plt.scatter(X, target_scaler.inverse_transform([df['Salary']]).reshape(-1))
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()
    
if __name__ == '__main__':
    main()