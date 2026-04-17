import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from matplotlib import pyplot as plt
from kneed import KneeLocator


def elbow_detection(inertias: list) -> int:
    """Calculates the Optimal Number of Clusters (k) using the Elbow method."""
    k_range = range(1, len(inertias) + 1)

    knee = KneeLocator(k_range, inertias, curve="convex", direction="decreasing")
    clusters = knee.knee

    plt.plot(k_range, inertias, marker="o")
    plt.title("Elbow graphic for Housing dataset")
    plt.xlabel("K (number of Clusters)")
    plt.ylabel("Inertia")
    plt.scatter(clusters, inertias[clusters - 1], color='red', s=100, zorder=5)
    plt.savefig("modules/clustering/elbow/housingData.png")

    return clusters


def main():
    df = pd.read_csv('data/HousingData.csv', sep=',', decimal='.')


    chas_imputer = SimpleImputer(strategy='most_frequent')
    df['CHAS'] = chas_imputer.fit_transform(df[['CHAS']])

    numerical_cols = df.drop(columns=['CHAS']).columns
    num_imputer = SimpleImputer(strategy='mean')
    df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

    encoder = OneHotEncoder(sparse_output=False)
    encoded_array = encoder.fit_transform(df[['CHAS']])
    encoded_df = pd.DataFrame(
        encoded_array,
        columns=encoder.get_feature_names_out(),
        index=df.index,
    )

    scaler = MinMaxScaler()
    scaled_array = scaler.fit_transform(df[numerical_cols])
    scaled_df = pd.DataFrame(scaled_array, columns=numerical_cols, index=df.index)

    normalized_df = pd.concat([encoded_df, scaled_df], axis=1)

    print(normalized_df)

    K = range(1, 300)
    inertias = []

    for i in K:
        model = KMeans(n_clusters=i, random_state=1, )
        model.fit(normalized_df)
        inertias.append(model.inertia_)

    optimal_clusters = elbow_detection(inertias)
    print(f"Optimal number of clusters: {optimal_clusters}")


if __name__ == "__main__":
    main()