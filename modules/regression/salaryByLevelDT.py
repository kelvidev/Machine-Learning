import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from utils.Normalizer import Normalizer
def main():

    df = pd.read_csv('data/Position_Salaries.csv')
    
    model = DecisionTreeRegressor(max_depth=8, criterion='squared_error')
    
    model.fit(df['Level'].to_numpy().reshape(-1,1), df['Salary'].to_numpy().reshape(-1,1))
    
    
    print(model.predict([[6.5]]))
    plt.plot(np.linspace(df['Level'].min(), df['Level'].max(), num=50), model.predict(np.linspace(df['Level'].min(), df['Level'].max(), num=50).reshape(-1,1)))
    plt.scatter(df['Level'], df['Salary'])
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()
    
if __name__ == '__main__':
    main()