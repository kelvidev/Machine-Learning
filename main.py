from utils.Normalizer import Normalizer 
import pandas as pd
from modules.clustering.Clusterer import Clusterer
from models.FootballPlayer import FootballPlayer 

from models.IrisFlower import IrisFlower
def testNormalizer():
    normalizer = Normalizer()
    normalizer.normalizeAll("data/dados_normalizar.csv")
    
    print(normalizer.denormalizeAll())    
    print(normalizer.normalizeAll())

    
def testClustering():
    
    
    
    iris_clusterer = Clusterer()
    iris_clusterer.findClusters("data/iris.csv")
    flower = IrisFlower.from_json(
        {
            "sepal_length":6.1,
            "sepal_width": 3.0,
            "petal_length": 4.6,
            "petal_width":1.4,
            "class": "Iris-versicolor",
        }
    )
    print(iris_clusterer.classifyInstance(flower.to_dataframe()))

    player_clusterer = Clusterer(separator=',')
    player_clusterer.findClusters("data/Players_avg_statistics.csv")

    player = FootballPlayer.from_json({
        "player": "caça rato",
        "shots_on_goal": 2.35,
        "disarms": 1.4,
    })
    print(player_clusterer.classifyInstance(player.to_dataframe()))
    

def main():
    # testNormalizer()
    testClustering()
    
    
if __name__ == "__main__":
    main()

