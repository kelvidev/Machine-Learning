from sklearn.cluster import KMeans
from utils.Normalizer import Normalizer
from matplotlib import pyplot as plt
from kneed import KneeLocator

"""Cauculates the Optimal Number of Clusters (k) """
def elbow_detection(inertias) -> int:
    
    knee = KneeLocator(range(1, 10), inertias, curve="convex", direction="decreasing")
    clusters = knee.knee
    
    plt.plot(range(1,10), inertias, marker="o")
    plt.title("Elbow graphic for Iris dataset")
    plt.xlabel("K(number of Clusters)")
    plt.ylabel("Inertia")
    plt.scatter(clusters, inertias[clusters], color='red', s=100, zorder=5, )
    plt.savefig("clustering/elbow/iris-elbow.png")
    
    return clusters


def main():
    print("--------------------------------------------------------------")
    
    normalizer = Normalizer()
    
    dataframe = normalizer.normalizeAll("data/iris.csv")
    
    inertias = []
    for k in range(1,10):
        model = KMeans(n_clusters=k, n_init=10, random_state=1)
        
        model.fit(dataframe)
        
        inertias.append(model.inertia_)

    clusters_num = elbow_detection(inertias)
    
    model = KMeans(n_clusters=clusters_num, n_init=10, random_state=1)
    model.fit(dataframe)
    dataframe = normalizer.denormalizeAll()
    dataframe['cluster'] = model.labels_
    print(dataframe.groupby(['cluster', 'class']).mean(numeric_only=True).round(2))
    
    
    
if __name__ == "__main__":
    main()
