from utils.Normalizer import Normalizer 
import pandas as pd

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
    
def main():
    testNormalizer()
    
    
if __name__ == "__main__":
    main()

