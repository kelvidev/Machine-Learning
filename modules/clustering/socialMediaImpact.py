import pandas as pd
import numpy as np 
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
import seaborn as sns
import matplotlib.pyplot as plt
from kneed import KneeLocator


def exploratory_data_analisys(df:pd.DataFrame):
    numerical_cols = df.select_dtypes(include=['number']).columns.to_list()
    categorical_cols = df.select_dtypes(include=['str']).columns.to_list()
    numerical_df = df[['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']]
    print(f'Numeric columns: {numerical_cols}')
    print(f'Categorical columns: {categorical_cols}')
    print(f'Total missing values: \n{df.isnull().sum()}')
    print(f'{numerical_df.describe().T}')
    
    
    # It gonna be goddam important sometime
    correlation = numerical_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation,
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        vmin=-1, vmax=1,
        square=True
    )
    plt.title("Correlation matrix")
    plt.tight_layout()
    plt.show()
    
"""Cauculates the Optimal Number of Clusters (k) """
def elbow_detection(inertias) -> int:
    
    knee = KneeLocator(range(1, 11), inertias, curve="convex", direction="decreasing")
    cluster = knee.knee
    
    plt.plot(range(1,11), inertias, marker="o" )
    plt.scatter(x=cluster, y=inertias[cluster],color='red', s=100, zorder=5)
    
    plt.title("players dataset inertias by n° of clusters")
    plt.xlabel("(k) n° of clusters")
    plt.ylabel("inertia")
    plt.savefig("modules/clustering/elbow/social_media_elbow.png")
    
    return cluster

def main():
    df = pd.read_csv('data/Social_media_impact_on_life.csv')

    print(df['Academic_Level'].nunique())

    exploratory_data_analisys(df)
    
    scaler = MinMaxScaler()
    num_df = df[['Age', 'Avg_Daily_Usage_Hours', 'Sleep_Hours_Per_Night', 'Mental_Health_Score']]
    num_array = scaler.fit_transform(num_df)
    numerical_df = pd.DataFrame(
        num_array,
        columns=num_df.columns,
        index=num_df.index
    )
    print(numerical_df)
    ordinal_encoder = OrdinalEncoder(categories=[['High School', 'Undergraduate', 'Graduate']])
    ord_df = df[['Academic_Level']]
    ord_array = ordinal_encoder.fit_transform(ord_df)
    ordinal_df = pd.DataFrame(
        ord_array,
        columns=ord_df.columns,
        index=ord_df.index,
    )    
    print(ordinal_df)
    
    categorial_encoder = OneHotEncoder(sparse_output=False)
    cat_df = df[['Gender', 'Country', 'Most_Used_Platform', 'Affects_Academic_Performance', ]]
    cat_array = categorial_encoder.fit_transform(cat_df)
    categorical_df = pd.DataFrame(
        cat_array,
        columns= categorial_encoder.get_feature_names_out(cat_df.columns)
    )
    norm_df = pd.concat([numerical_df, ordinal_df, categorical_df], axis=1,)
    
    inertias = []
    for i in range (1,11):
        model = KMeans(n_clusters=i,n_init=10, random_state=1)
        model.fit(norm_df)
        inertias.append(model.inertia_)
    
    clusters_num = elbow_detection(inertias)
    model = KMeans(n_clusters=clusters_num,n_init=10, random_state=1)
    model.fit(norm_df)
    
    df['cluster'] = model.labels_
    
    
if __name__ == '__main__':
    main()