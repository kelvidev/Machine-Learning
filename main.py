from Normalizer import Normalizer 

def main():
    normalizer = Normalizer()
    normalizer.normalize("data/haboski.csv")
    print(normalizer.denormalizeAll())    
    
    print(normalizer.normalize())
    
if __name__ == "__main__":
    main()

