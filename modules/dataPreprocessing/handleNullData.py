# Importing the necessary libraries
import pandas as pd;
from sklearn.impute import SimpleImputer;
import numpy as np;

dataset = pd.read_csv('data/pima-indians-diabetes.csv')

x = dataset.iloc[:,:-1].to_numpy()

missing = dataset.isnull().sum()

print(missing)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

imputer.fit(x[:,:])

x[:,:] = imputer.transform(x[:,:])

print(x)