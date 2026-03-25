from utils.Normalizer import Normalizer 
import pandas as pd
from clustering.Clusterer import Clusterer
def testNormalizer():
    normalizer = Normalizer()
    normalizer.normalizeAll("data/dados_normalizar.csv")
    
    print(normalizer.denormalizeAll())    
    print(normalizer.normalizeAll())
    
    new_instance = pd.DataFrame({"sexo":["M"]})
    new_instance2 = pd.DataFrame({"sexo":["F"]})
    triyng_to_break = pd.DataFrame({"sexo":["Macaco"]})
    
    print(normalizer.normalizeNominal(new_instance))
    print(normalizer.normalizeNominal(new_instance2))
    print(normalizer.normalizeNominal(triyng_to_break))
    
def testClustering():
    clusterer = Clusterer(separator=',')
    clusterer.findClusters("data/Players_avg_statistics.csv")

def main():
    testNormalizer()
    testClustering()
    
    
if __name__ == "__main__":
    main()

