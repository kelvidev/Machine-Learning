import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np

class Normalizer:

    def __init__(self, dataset=None, separator = ";", decimal = ".", ordinalData:list[map] = None) -> None:
        self.dataset = dataset
        self.dataframe = None
        self.encoder = OneHotEncoder(sparse_output=False)
        self.scaler = MinMaxScaler()
        self.categorical_cols_normalized = None
        self.categorical_cols_nonnormalized= None
        self.ordinalData = ordinalData
        self.numeric_cols = None
        self.textual_cols = None
        self.separator = separator
        self.decimal = decimal
        

    def findCategoricalData(self, df: pd.DataFrame, maxVariation: int = 5) ->list[str]:

        categoricalData = []
        for column in df.columns:
            unique_count = df[column].nunique()
            if unique_count <= maxVariation and not pd.api.types.is_numeric_dtype(df[column]) :
                categoricalData.append(column)  
        return categoricalData
    
    def findTextualData(self,  df: pd.DataFrame) -> list[str]:
        nonNumericData = []
        for column in df.columns:
            if not pd.api.types.is_numeric_dtype(df[column]):
                nonNumericData.append(column)
        return nonNumericData
                 
    def normalizeAll(self, dataset=None) -> pd.DataFrame:
        if(dataset is None):
            dataset = self.dataset
        else:
            self.dataset = dataset
        
        df = pd.read_csv(dataset, sep=self.separator, decimal=self.decimal)
        self.dataframe = df

        categorical_cols = self.findCategoricalData(df)
        self.categorical_cols_nonnormalized = categorical_cols
        textual_cols = self.findTextualData(df)
        numeric_cols = [col for col in df.columns if col not in categorical_cols and col not in textual_cols]
        for item in categorical_cols:
            if item  in textual_cols:
                textual_cols.remove(item)
        self.textual_cols = textual_cols
        categorical_nd_array = self.encoder.fit_transform(df[categorical_cols])
        categorical_df = pd.DataFrame(
            categorical_nd_array, 
            columns= self.encoder.get_feature_names_out(categorical_cols),
            index= df.index
        )
        self.categorical_cols_normalized = [col for col in categorical_df.columns]
        
        numeric_nd_array = self.scaler.fit_transform(df[numeric_cols])
        numeric_df = pd.DataFrame(
            numeric_nd_array,
            columns= numeric_cols,
            index=df.index,
        )
        self.numeric_cols = [col for col in numeric_df.columns]
        
        complete_df = pd.concat([categorical_df, numeric_df, df[textual_cols]], axis=1)
        
        self.dataframe = complete_df
        return complete_df
                
    def denormalizeAll(self, instance:pd.DataFrame=None) -> pd.DataFrame :
        if(instance is None):
            dataframe = self.dataframe.copy()
        else:
            dataframe = instance
            
        dataframe_parts = []
        
        if(self.categorical_cols_normalized):
            decoded_categorical_nd_array = self.encoder.inverse_transform(dataframe[self.categorical_cols_normalized])
            decoded_categorical_df = pd.DataFrame(decoded_categorical_nd_array, columns= self.encoder.feature_names_in_)
            dataframe_parts.append(decoded_categorical_df)
                    
        if(self.numeric_cols):
            decoded_numeric_nd_array = self.scaler.inverse_transform(dataframe[ self.numeric_cols])    
            decoded_numeric_df = pd.DataFrame(decoded_numeric_nd_array, columns= self.scaler.feature_names_in_)
            dataframe_parts.append(decoded_numeric_df)
        
        if(self.textual_cols):
            dataframe_parts.append(dataframe[self.textual_cols])
        
        dataframe = pd.concat(dataframe_parts, axis=1) 
        if(instance==None):
            self.dataframe = dataframe
        
        return dataframe
    
    def normalizeInstance(self, instance: pd.DataFrame) -> pd.DataFrame:
        
        try:
            categorical_cols = [item for item in self.findTextualData(instance) if item not in self.textual_cols] 
            categorical_nd_array = self.encoder.transform(instance[categorical_cols])
            categorical_df = pd.DataFrame(
                categorical_nd_array, 
                columns=self.categorical_cols_normalized,
                index= instance.index
            )
            
            numeric_nd_array = self.scaler.transform(instance[self.numeric_cols])
            numeric_df = pd.DataFrame(
                numeric_nd_array,
                columns= self.numeric_cols,
                index=instance.index,
            )
            complete_df = pd.concat([categorical_df, numeric_df, instance[self.textual_cols]], axis=1)
            return complete_df
        except Exception as error:
            print(error)
            return "that value could not be normalized"
        
    def getTrainableDataFrame(self)-> pd.DataFrame:
        return self.dataframe[self.categorical_cols_normalized + self.numeric_cols]