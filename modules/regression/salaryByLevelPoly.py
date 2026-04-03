import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import pandas as pd
from utils.Normalizer import Normalizer
def main():

    df = pd.read_csv('data/Position_Salaries.csv')

    poly = PolynomialFeatures(degree=4)
    level_poly = poly.fit_transform(df['Level'].to_numpy().reshape(-1,1))
    
    model = LinearRegression()

    model.fit(level_poly, df['Salary'].to_numpy().reshape(-1,1))

    salarys = model.predict(poly.transform(df['Level'].to_numpy().reshape(-1,1)))

    print(model.predict(poly.transform([[6.5]])))
    plt.plot(df['Level'], salarys)
    plt.scatter(df['Level'], df['Salary'])
    plt.ticklabel_format(style='plain', axis='y')
    plt.show()
    
if __name__ == '__main__':
    main()