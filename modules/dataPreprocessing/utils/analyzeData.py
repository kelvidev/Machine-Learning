import pandas as pd

def checkDataVariation(dataset: pd.DataFrame):
    for column in dataset.columns:
        print(f"{column}: {dataset[column].nunique()} unique values")
    return;

def findCategoricalData(dataset: pd.DataFrame, maxVariation: int = 5):
    for column in dataset.columns:
        unique_count = dataset[column].nunique()
        if unique_count <= maxVariation:  
            print(f"{column}: {unique_count} unique values", end="")
            print(f"  → {dataset[column].unique().tolist()}", )
    print("No more Categorical Data")
    return;

def checkUniqueByColumn(dataset: pd.DataFrame, column: str):
    unique_values = dataset[column].unique()
    print(f"  Unique values for {column} ({len(unique_values)}): {unique_values}")
    return;






