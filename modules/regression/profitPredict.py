from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import numpy as np
from utils.analyzer import save_feature_previews
def main():
    print('--------------------------------------------------\n')
    dataframe = pd.read_csv("data/startups.csv")
    
    X = dataframe.drop('Profit', axis=1)
    y = dataframe['Profit']
    
    save_feature_previews(X,y, 'modules/regression/dataBehavior/preview/profit-predict')
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42 )    

    feature_encoder = OneHotEncoder(sparse_output=False)
    
    categorical_array = feature_encoder.fit_transform(X_train[['State']])
    categorical_df = pd.DataFrame(
        categorical_array,
        columns=feature_encoder.get_feature_names_out(['State']),
        index=X_train.index,
    )    
    
    X_train = pd.concat([X_train , categorical_df], axis=1).drop('State', axis=1)
    
    categorical_test_array = feature_encoder.transform(X_test[['State']])
    categorical_test_df = pd.DataFrame(
        categorical_test_array,
        columns=feature_encoder.get_feature_names_out(['State']),
        index=X_test.index,
    )    
    
    X_test = pd.concat([X_test , categorical_test_df], axis=1).drop('State', axis=1)
    
    model = LinearRegression()
    
    model.fit(X=X_train, y=y_train)
    
    predictec_y = model.predict(X=X_test)
    
    np.set_printoptions(precision=2)
    print(np.concatenate((predictec_y.reshape(-1,1), y_test.to_numpy().reshape(-1,1)), axis=1))
    
    print(f"The accuracy was {(r2_score(y_pred= predictec_y  , y_true=y_test) * 100):.2f}%" )
    
if __name__ == "__main__":
    main()