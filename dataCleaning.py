from sklearn.preprocessing import OneHotEncoder
import numpy as np

cores = [['Vermelho'], ['Azul'], ['Verde'], ['Azul']]

encoder = OneHotEncoder(sparse_output=False) 
cores_codificadas = encoder.fit_transform(cores)

print("Categorias originais:", [c[0] for c in cores])
print("Categorias encontradas:", encoder.categories_)
print("Codificação:\n", cores_codificadas)
print("Colunas geradas:", encoder.get_feature_names_out(['Cor']))