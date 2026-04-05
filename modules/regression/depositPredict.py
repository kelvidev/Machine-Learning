from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


def main():
    
    df = pd.read_csv('data/bank-full.csv', sep=';')
    
    df = df[df['education'] != 'unknown']
    
    X = df.drop(columns=['y'])
    y = df['y']
    
    transformer = ColumnTransformer(
        [
            (
                'ordinalEncoder',
                OrdinalEncoder(
                    categories=[['primary', 'secondary', 'tertiary']]
                ),
                ['education'],
            ),
            (
                'oneHotEncoder',
                OneHotEncoder(sparse_output=False),
                ['job', 'marital', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
            )
        ],
        remainder='passthrough',
    )
    
    X = transformer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
    
    no_indices = np.where(y_train == 'no')[0]
    keep_no = np.random.choice(no_indices, size=int(len(no_indices) * 0.3), replace=False)

    yes_indices = np.where(y_train == 'yes')[0]
    keep_indices = np.sort(np.concatenate([keep_no, yes_indices]))

    X_train = X_train[keep_indices]
    y_train = y_train.iloc[keep_indices]
    
    model = RandomForestClassifier(n_estimators=150, random_state=1, class_weight='balanced')

    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)

    f1_per_class = classification_report(y_pred=y_pred, y_true=y_test)
    print(f1_per_class)
    # print(f'\n{(score * 100):.2f}%')    

if __name__ == '__main__':
    main()