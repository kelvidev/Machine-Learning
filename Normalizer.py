import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
class Normalizer:

    def __init__(self, dataset=None) -> None:
        self.dataset = dataset

    def findCategoricalData(self, dataset: pd.DataFrame, maxVariation: int = 5):
        '''
        the intention is to request an AI API to know if the found categorical data makes sense
        '''
        categoricalData = []
        for column in dataset.columns:
            unique_count = dataset[column].nunique()
            if unique_count <= maxVariation:
                categoricalData.append(column)  
                print(f"{column}: {unique_count} unique values", end="")
                print(f"  → {dataset[column].unique().tolist()}", )
        print("No more Categorical Data")
        return categoricalData
    
    def findTextualData(self,  dataframe: pd.DataFrame):
        nonNumericDAta = []
        for column in dataframe.columns:
            if not pd.api.types.is_numeric_dtype(dataframe[column]):
                nonNumericDAta.append(column)
        return nonNumericDAta
                
        
    def normalize(self, dataset=None):
        if(dataset is None):
            dataset = self.dataset
        dataframe = pd.read_csv(dataset, sep=';', decimal=',')

        listOfCategorical = self.findTextualData(dataframe)
        
        categorical = dataframe[listOfCategorical]
        nonCategorical = dataframe.drop(columns=listOfCategorical)
        # try:
        #     noCategorical = noCategorical.drop(columns= self.findTextualData(dataframe))
        # except KeyError as e:
        #     print("Column was already deleted", e)
        print(nonCategorical)
        print(categorical)
        
        encoder = OneHotEncoder()
        
        scaler = MinMaxScaler()
        categorical = encoder.fit_transform(categorical)
        
        nonCategorical = scaler.fit_transform(nonCategorical)
        
        print(categorical)
        print(nonCategorical)
                
        
