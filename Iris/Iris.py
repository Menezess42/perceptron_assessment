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
            self.weights = np.random.rand(self.numberOfPerceptrons * (self.numberOfInputs + 1))

    def save_model(self, file_path: str):
        '''
        salva o modelo em um arquvio JSON.
        '''
        model_data = {
                "numberOfPerceptrons": self.numberOfPerceptrons,
                "numberOfInputs": self.numberOfInputs,
                "numberOfOutputs": self.numberOfOutputs,
                "bias": self.bias,
                "weights": self.weights.tolist()
                }
        with open(file_path, 'w') as f:
            json.dump(model_data, f, indent=4)
        print(f"Modelo salvo em {file_path}")

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

    def divid_data(self,
                   perTrain: float = 0.7,
                   perTest: float = 0.15,
                   perVal: float = 0.0, data=[]) -> tuple:

        if perVal == 0.0:
            perVal = 1 - (perTrain+perTest)
        data_size = len(data)
        test_size = int(data_size*perTest)
        val_size = int(data_size*perVal)
        train_size = int(data_size*perTrain)
        test = data[:test_size]
        val = data[test_size:test_size+val_size]
        train = data[test_size+val_size:]
        return train, val, test

    def softMax(self, z: np.ndarray) -> np.ndarray:
        '''
        Aplica a função softmax sobre o vetor de saídas
        z dos perceptrons.
        Z: vetor numpy com as ativações dos perceptrons
        (shape: [numberOfPerceptrons])
        Retorna: Vetor com as probabilidades normalizadas
        (shape: [numberOfPerceptrons])
        '''
        z = z.astype(np.float64)
        z_stable = z - np.max(z)
        exp_z = np.exp(z_stable)
        softmax = exp_z / np.sum(exp_z)
        return softmax


    def main(self, file_path: str):
        '''
        file_path: Caminho dos dados
        '''
        raw_data = open_data(file_path)
        data = self.from_raw_to_matrix(raw_data)
        hoted_data, unique_cats = self.one_hotEncoder(data)
        shuffled_data = self.shuffle(hoted_data)
        train, val, test = self.divid_data(perTrain=0.7, perTest=0.15, data=shuffled_data)
        self.save_model('./Reports/model_b4_train.json')
        self.train(train)

    def train(self,
              train,
              learning_rate=0.01,
              epochs=100):
        '''
        Treina a rede usando gradiente descendente.
        train: dados de treinamento (entrada + saída one-hot)
        '''
        train = np.array(train)
        for epoch in range(epochs):
            total_loss = 0
            for row in train:
                *x, y_true = row[:-self.numberOfPerceptrons], row[-self.numberOfPerceptrons:]
                x = np.array(x, dtype=float)
                y_true = np.array(y_true, dtype=int)
                
                # Forward
                y_pred = self.forward(x)

                # Perda (entropia cruzada)
                loss = -np.sum(y_true*np.log(y_pred+1e-15))
                total_loss += loss

                # Gradiente da softmax com entropia cruzada
                grad = y_pred - y_true # shape: [numberOfPerceptrons]

                # prepara x com bias
                x_bias = np.append(x, self.bias) # shape: [numberOfInputs + 1]

                # Atualiza os pesos para cada perceptron
                for i in range(self.numberOfPerceptrons):
                    idx_start = i*(self.numberOfInputs+1)
                    idx_end = idx_start + self.numberOfInputs+1
                    self.weights[idx_start:idx_end] -= learning_rate * grad[i] * x_bias

            # Epoch feddback
            average_loss = total_loss/len(train)
            print(f'Época {epoch+1}/{epochs} - perda total: {total_loss:.4f} - Perda média: {average_loss:.4f}')

        # Salva o modelo após o treinamento
        save_path = './Reports/model.json'
        self.save_model(save_path)

    
    def test(self, test):
        ...

    def valid(self, valid):
        ...
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        '''
        Calcula a saída da rede (logists)+ aplica softmax.
        x: vetor de entrada (shape: [numberOfInputs])
        retorna: vetor de probabilidades (shape: [numberOfPerceptrons])
        '''
        # Adiciona o bias como entrada extra
        x = np.append(x, self.bias)

        # Grante que os pesos estão reshaped corretamente
        weights_matrix = self.weights.reshape(self.numberOfPerceptrons, self.numberOfInputs + 1)

        # Produto escalar de cadad neurônio com os inputs
        logits = np.dot(weights_matrix, x)

        # Softmax para converter logits em probabilidades
        returns = self.softMax(logits)
        return returns
    

def main():
    pn = PerceptronNetwork(numberOfPerceptrons=3,
                           numberOfInputs=4,
                           numberOfOutputs=3)
    pn.main(file_path='./DataSet/iris.data')


if __name__ == "__main__":
    main()




















