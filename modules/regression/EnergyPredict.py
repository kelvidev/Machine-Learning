import pandas as pd
from sklearn.linear_model import LinearRegression 
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from utils.analyzer import save_feature_previews

def main():
    
    df = pd.read_csv('data/Energy-output.csv', sep=',', decimal='.')
    
    X = df.drop(columns=['PE'])
    y = df['PE']
    
    save_feature_previews(X,y, 'modules/regression/dataBehavior/preview/energy-predict')
       
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    linearRegressor = LinearRegression()
    linearRegressor.fit(X_train, y_train)
    score = r2_score(y_pred=linearRegressor.predict(X_test), y_true=y_test)    
    print(f'\n Linear Regressor score was {(score * 100):.2f}%')

 
if __name__ == '__main__':
    main()