# Importing the necessary libraries
import numpy as np
import utils.analyzeData as analyzer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder;
from sklearn.compose import ColumnTransformer;
import pandas as pd;

dataset = pd.read_csv('data/titanic.csv')

analyzer.findCategoricalData(dataset);

X = dataset.drop(['Survived',], axis=1);
y = dataset['Survived'].to_numpy();

ct = ColumnTransformer(transformers=[
    (
        'onehot',
        OneHotEncoder(),
        ["Embarked", "Sex", "Pclass"]
    )
], remainder='passthrough')

encoded_data =np.array( ct.fit_transform(X))

'''

le = LabelEncoder()
y = le.fit_transform(y)

i dont gonna do it because i dont have to 
 - Muhammad Ali 

'''

print(encoded_data)
print(y)