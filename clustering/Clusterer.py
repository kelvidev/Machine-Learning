from sklearn.cluster import KMeans
from utils.Normalizer import Normalizer
from matplotlib import pyplot as plt
from kneed import KneeLocator
import pandas as pd
class Clusterer:
    
    def __init__(self, dataset:str=None, separator=";", decimal="."):
        self.dataset = dataset
        self.name = None
        self.separator = separator
        self.decimal = decimal
        
    """Cauculates the Optimal Number of Clusters (k) """
    def elbow_detection(self,inertias) -> int:
        
        knee = KneeLocator(range(1, 10), inertias, curve="convex", direction="decreasing")
        cluster = knee.knee
        
        plt.plot(range(1,10), inertias, marker="o")
        plt.title(f"Elbow graphic for {self.name} dataset")
        plt.xlabel("K(number of Clusters)")
        plt.ylabel("Inertia")
        plt.scatter(cluster, inertias[cluster], color='red', s=100, zorder=5, )
        plt.savefig(f"clustering/elbow/{self.name}-elbow.png")
        
        return cluster


    def findClusters(self, dataset:str=None)-> pd.DataFrame:
        
        if(dataset is None):
            dataset = self.dataset
        else:
            self.dataset = dataset
        self.name = self.dataset.split(sep=".")[0].replace("data/", "")
        normalizer = Normalizer(separator=self.separator, decimal=self.decimal)
        
        normalizer.normalizeAll(self.dataset)
        train_dataframe = normalizer.getTrainableDataFrame()
        inertias = []
        for k in range(1,10):
            
            model = KMeans(n_clusters=k, n_init=10, random_state=1)
            model.fit(train_dataframe)
            inertias.append(model.inertia_)

        clusters_num = self.elbow_detection(inertias)
        
        model = KMeans(n_clusters=clusters_num, n_init=10, random_state=1)
        model.fit(train_dataframe)
        dataframe = normalizer.denormalizeAll()
        dataframe['cluster'] = model.labels_
        
        print(
            "it found the following cluster groups: \n",
            dataframe.groupby('cluster').mean(numeric_only=True).round(2),
            "\n",
        )
        
        print(dataframe)