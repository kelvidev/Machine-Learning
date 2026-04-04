import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from utils.Normalizer import Normalizer
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv('data/winequality-red.csv', sep=';', decimal='.')
    
    X = df.drop(columns=['quality'])
    y = df['quality']
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)
    
    rf_regressor = RandomForestClassifier(n_estimators=100, random_state=1,)
    rf_regressor.fit(X_train, y_train)
    
    rf_score = accuracy_score(y_pred=rf_regressor.predict(X_test), y_true=y_test)
    
    print(f'\n {rf_score * 100} %')
    
    print(rf_regressor.feature_importances_)
    
if __name__ == '__main__':
    main()