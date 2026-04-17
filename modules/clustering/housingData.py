import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import KNNImputer, SimpleImputer
from matplotlib import pyplot as plt
from kneed import KneeLocator
from scipy.spatial.distance import cdist


def __number_optimal_clusters(distorsions, k_range):
        distances_length = __calculate_distance_length(k_range, distorsions)
        distances = []
        for i in range(len(distorsions)):
            x = k_range[i]
            y = distorsions[i]

            brute_distance = __calculate_brute_distance(k_range,distorsions,x, y)
            
            distances.append(brute_distance/distances_length)

        number_optimal_clusters = k_range[distances.index(np.max(distances))]
        return number_optimal_clusters

def __calculate_brute_distance(k_range, distorsions, x, y):
        x0 = k_range[0]
        y0 = distorsions[0]
        xn = k_range[-1]
        yn = distorsions[-1]

        brute_distance = abs(((yn-y0)*x) - ((xn-x0)*y) + (xn*y0) - (yn*x0))

        return brute_distance
    
def __calculate_distance_length(k_range, distorsions):
    x0 = k_range[0]
    y0 = distorsions[0]
    xn = k_range[-1]
    yn = distorsions[-1]

    distance_length = np.sqrt(((yn-y0)**2) + ((xn-x0)**2))

    return distance_length
    
def __calculate_distorsion(cluster: KMeans, data):
        distance_centers = cdist(data, cluster.cluster_centers_, 'euclidean')
        minor_distance = np.min(distance_centers, axis=1)
        distorsion = sum(minor_distance) / data.shape[0]
        return distorsion

"""Cauculates the Optimal Number of Clusters (k) """
def elbow_detection(inertias) -> int:
    
    knee = KneeLocator(range(1, 101), inertias, curve="convex", direction="decreasing")
    clusters = knee.knee
    
    plt.plot(range(1,101), inertias, marker="o")
    plt.title("Elbow graphic for Iris dataset")
    plt.xlabel("K(number of Clusters)")
    plt.ylabel("Inertia")
    plt.scatter(clusters, inertias[clusters], color='red', s=100, zorder=5, )
    plt.savefig("modules/clustering/elbow/housingData.png")
    
    return clusters


df = pd.read_csv('data/HousingData.csv', sep=',', decimal='.')

imputer = KNNImputer(n_neighbors=5)

categorical_array = df['CHAS'].to_list()

for index in range(len(categorical_array)):
    if (categorical_array[index] != 1 and categorical_array[index] != 0):
        categorical_array[index] = 0
        
categorical_df = pd.DataFrame(categorical_array, columns=['CHAS'])

encoder = OneHotEncoder(sparse_output=False)

encoded_array = encoder.fit_transform(categorical_df)

print(encoded_array)

scaler = MinMaxScaler()

numerical_df = df.drop(columns=['CHAS'])


scaled_array = scaler.fit_transform(numerical_df)

scaled_df = pd.DataFrame(
    scaled_array,
    columns=numerical_df.columns,
    index= numerical_df.index
)


encoded_df = pd.DataFrame(
    encoded_array,
    columns= encoder.get_feature_names_out(),
    index=df.index,
)

normalized_df = pd.concat([encoded_df, scaled_df ], axis=1)

imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
print(normalized_df)
print("____________________depois________________--")
df = imputer.fit_transform(normalized_df)

normalized_df = pd.DataFrame(
    df,
    columns=normalized_df.columns,
    index=normalized_df.index,
)


distortions = []

K = range(1,300)

for i in K:
    model = KMeans(n_clusters=i, random_state=1, n_init=10,)

    model.fit(normalized_df)
    
    distortions.append(__calculate_distorsion(model, normalized_df))

clusters = __number_optimal_clusters(distortions, K)    

print(normalized_df)








