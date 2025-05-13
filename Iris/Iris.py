# Iris:

# 150 itens;
# # 50 Setosa;
# # 50 Virginica;
# # 50 Veriscolor;

# Para cada item foram obtidos 4 atributos:
# # Comprimento da sépala;
# # Comprimento da pétala;
# # Largura da sépala;
# # Largura da pétala;

# Objetivo:
# # Treinar uma rede Perceptron para reconhecer as três
# # diferentes classes;
# # Divida aleatorimaente os exemplos em subconjuntos de treinamento
# # # Recomendação Professor: 70% Treinamento 15% validação 15% teste;

# Testes:
# ># Rodar 100 vezes para cada experimento;
# ># Diferentes pesos de inicialização;
# # # Escolher um ponto mediano. Incrementar e decrementar este ponto a mesma
# # # #quantidade de vezes e coletar os dados;
# ># Diferentes taxas de aprendizdo;
# # # Escolher um ponto mediano. Incrementar e decrementar este ponto a mesma
# # # #quantidade de vezes e coletar os dados;
# ># Diferentes pesos de inicialização E taxas de aprendizdo;
# # # Combinação dos dois experimentos anteriores;
# ># Rodar todos os experimentos assima 1 vez com os dados normais e 1 vez com
# # os dados normalizados;

# Gráficos:
# # Gráfico de erro médio quadrático;
# # Entropia cruzada Categŕoica Média mostrando a convergência do algoritmo;
# # ...;
import numpy as np
import os
import json
import pandas as pd

def grante_pasta(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def carregar_json(path: str): 
    with open(path, 'r') as f:
        return json.load(f)


def open_data(path: str):
    with open(path, 'r') as f:
        return f.read()

class Perceptron:
    ...


class PerceptronNetwork:
    def __init__(self, numberOfPerceptrons: int,
                 numberOfInputs: int,
                 numberOfOutputs: int,
                 bias: int = 1,
                 weights=[]) -> None:
        '''
        numberOfPerceptrons: Numero de perceptrons
        numberOfInputs: numero de inputs
        numberOfOutputs: Numero de outputs
        bias: bias, default=1
        weights: Vetor de pesos dos inputs; Se deixado vazio é
        preenchido por valores vazios
        '''

        self.numberOfPerceptrons = numberOfPerceptrons
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.bias = bias

        if weights == []:
            self.weights = np.random.rand(numberOfInputs+1)

    def shuffle(self, data):
        np.random.shuffle(data)
        return data

    def one_hotEncoder(self, data):
        data = np.array(data, dtype=object)
        labels = data[:, -1]
        unique_cats = sorted(set(labels))
        cat_to_index = {cat: idx for idx, cat in enumerate(unique_cats)}

        one_hot = np.zeros((len(labels), len(unique_cats)), dtype=int)
        for i, label in enumerate(labels):
            col = cat_to_index[label]
            one_hot[i, col] = 1

        # Remove a coluna original e concatena one-hot
        data = np.hstack((data[:, :-1], one_hot))
        return data, unique_cats

    def from_raw_to_matrix(self, data: str) -> np.matrix:
        split1 = data.split('\n')
        a = []

        for item in split1:
            if not item.strip():
                continue
            parts = item.split(',')
            *floats, last = parts
            floats = [float(x) for x in floats]
            floats.append(last)
            a.append(floats)

        return a

    def main(self, file_path: str):
        '''
        file_path: Caminho dos dados
        '''
        raw_data = open_data(file_path)
        data = self.from_raw_to_matrix(raw_data)
        hoted_data, unique_cats = self.one_hotEncoder(data)
        shuffled_data = self.shuffle(hoted_data)


def main():
    pn = PerceptronNetwork(numberOfPerceptrons=3,
                           numberOfInputs=4,
                           numberOfOutputs=3)
    pn.main(file_path='./DataSet/iris.data')


if __name__ == "__main__":
    main()




















