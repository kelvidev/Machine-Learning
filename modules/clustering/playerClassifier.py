from utils.Normalizer import Normalizer
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from kneed import KneeLocator

"""Cauculates the Optimal Number of Clusters (k) """
def elbow_detection(inertias) -> int:
    
    knee = KneeLocator(range(1, 10), inertias, curve="convex", direction="decreasing")
    cluster = knee.knee
    
    plt.plot(range(1,10), inertias, marker="o" )
    plt.scatter(x=cluster, y=inertias[cluster],color='red', s=100, zorder=5)
    
    plt.title("players dataset inertias by n° of clusters")
    plt.xlabel("(k) n° of clustera")
    plt.ylabel("inertia")
    plt.savefig("modules/clustering/elbow/players-elbow.png")
    
    return cluster

def main():
    
    normalizer = Normalizer(separator=",")
    normalizer.normalizeAll("data/Players_avg_statistics.csv")
    dataframe = normalizer.dataframe
    
    inertias = []
    for k in range(1,10):
        model = KMeans(n_clusters=k,random_state=1, n_init=10,)
        model.fit(dataframe.drop(normalizer.textual_cols, axis=1))
        inertias.append(model.inertia_)
    
    clusters_num = elbow_detection(inertias=inertias)
    
    model = KMeans(n_clusters=clusters_num, n_init=10, random_state=1)
    model.fit(dataframe.drop(normalizer.textual_cols, axis=1))
    
    dataframe = normalizer.denormalizeAll()
    dataframe["cluster"] = model.labels_
    
    print(
        "it found the following groups and avg: \n",
        dataframe.groupby('cluster').mean(numeric_only=True).round(2),
        "\n",
    )
    print(dataframe)
        
if __name__ == "__main__":
    main()
