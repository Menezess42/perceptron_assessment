import os
import json
import numpy as np
import pandas as pd

def garante_pasta(path: str) -> None:
    """
    path: str - Caminho da pasta.
    Função responsável por garantir a existencia das pastas.
    """
    # Verifica se a pasta existe
    if not os.path.exists(path):
        # Se não existe então cria
        os.makedirs(path)

def carregar_json(path: str):
    """
    Função responsável por abrir arquivos json
    """
    with open(path, "r") as f:
        return json.load(f)

def open_data(path: str):
    """
    Função responsável por abrir os arquivos .data,
    neste caso usada para abrir o Iris.data
    """
    with open(path, "r") as f:
        return f.read()

class PerceptronNetwork:
    """
    Classe principal. Responsável por criar a rede perseptron e realizar os testes no dataSet
    """
    def __init__(self, numberOfPerceptrons: int, numberOfInputs: int, numberOfOutputs: int, bias: int = 1, weights=[]) -> None:
        """
        numberOfPerceptrons: Numero de perceptrons
        numberOfInputs: Numero de inputs
        numberOfOutputs: Numero de outputs
        bias: Bias, default=1 weights: Vetor de pesos dos inputs; Se deixado vazio é preenchido por valores vazios
        """
        self.numberOfPerceptrons = numberOfPerceptrons
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.bias = bias

        if weights == []:
            self.weights = np.random.rand(self.numberOfPerceptrons * (self.numberOfInputs + 1))

    def from_raw_to_matrix(self, data: str) -> np.ndarray:
        """
        data: DataSet vindo direto da leitura Wine.data
        Função responsável por ler o Wine.data, que contém os dados separados por virgulas
        e as linhas separados por \n e converte para dados em uma array.
        """
        lines = data.strip().split("\n") 
        # Remove espaços em branco no início/fim da string e divide a string em várias linhas
        # Cada elemento da lista "lines" representa uma linha do dataset

        dataset = []
        # Lista que irá conter os dados já convertidos

        for line in lines: # Para cada linha em linhas
            if line.strip():
                # Ignora linhas em brancos

                parts = line.split(',')
                # Divide a linha pelos separadores de vírgula
                # O primeiro elemento é a classe, os demais são as features

                cls = int(parts[0])
                # Converte a classe (rótulo) para inteiro

                features = [float(x) for x in parts[1:]]
                # Converte as features restantes para float

                features.append(cls)
                # coloca a classe como último elemento da linha (útil para classificação supervisionada)
                dataset.append(features)
                # Adiciona a linha convertida à lista principal
        return np.array(dataset, dtype=object) # Converte a lista final para uma array NumPy

    def normalize_between_minus1_1(self, data: np.ndarray) -> np.ndarray:
        """
        Normaliza os dados para o intervalo [-1, 1].
        Parâmetros:
        data: ndarray onde cada linha representa uma amostra e a última coluna é a classe (não será normalizado)
        """
        features = data[:, :-1].astype(float)
        # Separa todas as colunas menos a útlima (que são os rótulos)
        # Converte para float para garantir que operações matemáticas funcionem corretamente
        min_vals = features.min(axis=0)
        # Calcula o valor mínimo de cada feature (coluna)
        max_vals = features.max(axis=0)
        # Calcula o valor máximo de cada feature (coluna)
        norm_features = 2 * (features - min_vals) / (max_vals - min_vals) - 1
        # Aplica a normalização para o intervalo [-1, 1] usando a fórmula:
        # x_norm = 2 * (x-min) / (max - min) - 1
        data[:, :-1] = norm_features
        # Substitui os valores originais das features pelos valores normalizados
        # A última coluna (classe) permanece intacta
        return data

    def one_hotEncoder(self, data: np.ndarray):
        """
        Converte a última coluna de rótulos do dataset em codificação one-hot.

        Parâmetros:
        data: ndarray a última coluna contém os rótulos (classes)
        """
        labels = data[:, -1] # Pega a última coluna do array, que contém os rótulos (classes)
        unique_cats = sorted(set(labels)) # Obtém as categorias únicas (rótulos distintos), ordenadas
        cat_to_index = {cat: idx for idx, cat in enumerate(unique_cats)} # Cria um dicionário que mapeia cada categoria para um índice (posição na codificação one-hot)
        one_hot = np.zeros((len(labels), len(unique_cats)), dtype=int)
        # Cria uma matriz de zeros para armazenar os vetores one-hot
        # Cada linha corresponde a uma amostra, cada coluna a uma categoria 
        for i, label in enumerate(labels):
            one_hot[i, cat_to_index[label]] = 1
            # Marca com 1 a posição correspondente à classe daquela amostra
        return np.hstack((data[:, :-1], one_hot)), unique_cats

    def shuffle(self, data):
        np.random.shuffle(data) # Embaralha os dados
        return data

    def divid_data(self, perTrain=0.7, perTest=0.15, perVal=0.15, data=[]):
        # Divide os dados em conjuntos de treino, validação e teste.
        if perVal == 0.0:
            perVal = 1 - (perTrain + perTest)
            # Caso o usuário não defina perVal, calcula o restante da divisão 
        data_size = len(data)
        # Tamanho total dos dados
        test_size = int(data_size * perTest) 
        val_size = int(data_size * perVal)
        train_size = data_size - test_size - val_size
        # Calcula o número de amostras para cada conjunto
        test = data[:test_size]
        val = data[test_size:test_size + val_size]
        train = data[test_size + val_size:]
        # Divide os dados nas três partes
        return train, val, test

    def softMax(self, z: np.ndarray) -> np.ndarray:
        # Aplica a função softmax às ativiações dos perceptrons
        z = z.astype(np.float64)
        # Garante que os valores são float64 para maior estabilidade númerica
        z_stable = z - np.max(z)
        # Subtrai o valor máximo de z para evitar overflow numérico ao aplicar exp()
        exp_z = np.exp(z_stable)
        # Aplica a exponencial em cada valor de z
        return exp_z / np.sum(exp_z)

    def forward(self, x: np.ndarray) -> np.ndarray:
        # Executa o passo de iferência (forward pass)
        x = np.append(x, self.bias)
        # Adiciona o bias ao final do vetor de entrada
        weights_matrix = self.weights.reshape(self.numberOfPerceptrons, self.numberOfInputs + 1)
        # Reshape dos pesos para uma matriz onde cada linha é o vetor de pesos de um perceptron
        logits = np.dot(weights_matrix, x)
        # calcula o produto escalar entre entradas e pesos para obter os logits
        return self.softMax(logits)

    def cross_entropy(self, y_true, y_pred) -> float:
        # Calcula a perda cross-entropy entre os rótulos reais e as predições 
        return -np.sum(y_true * np.log(y_pred + 1e-15))

    def softmax_cross_entropy_gradient(self, y_true, y_pred) -> np.ndarray:
        # Derivada da funçaõ de perda cross-entropy combinada com softmax
        return y_pred - y_true

    def valid(self, val_data):
        # Avalia o desempenho do modelo nos dados de validação
        val_data = np.array(val_data)
        total_loss = 0
        total_mse = 0
        for row in val_data:
            *x, y_true = row[: -self.numberOfPerceptrons], row[-self.numberOfPerceptrons :]
            # Separa as features (x) e os rótulos one-hot (y_true)
            x = np.array(x, dtype=float)
            y_true = np.array(y_true, dtype=int)
            y_pred = self.forward(x)
            # Executa o forward pass para obter a predição 
            total_loss += self.cross_entropy(y_true, y_pred)
            # Acumula a perda cross-entropy
            total_mse += np.mean((y_true - y_pred) ** 2)
            # Acumula a média do erro quadrático (MSE)
        return {
            "cross_entropy_loss": round(total_loss / len(val_data), 6),
            "mse_loss": round(total_mse / len(val_data), 6),
            "total_loss": round(total_loss, 6)
        }

    def save_model(self, file_path: str):
        """
        file_path: Caminho onde o modelo deve ser salvo
        Função que salva em um arquivo .json o modelo.
        """
        model_data = {
            "numberOfPerceptrons": self.numberOfPerceptrons,
            "numberOfInputs": self.numberOfInputs,
            "numberOfOutputs": self.numberOfOutputs,
            "bias": self.bias,
            "weights": self.weights.tolist()
        }
        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=4)

    def log_json(self, report_data: list[dict], file_path: str):
        """
        report_data | list[Dict]: dados para serem salvos.
        file_path | str: caminho na qual o log deve ser salvo.
        Função para salvar os logs.
        """
        garante_pasta(os.path.dirname(file_path))
        with open(file_path, "w") as f:
            json.dump(report_data, f, indent=4)

    def test(
        self,
        test_data,
        model_path: str,
        label_names=None,
        report_path="./Reports/test_report.json",
    ):
        """
        Avalia o modelo carregado com os dados de teste.

        test_data: Lista ou array com as amostras de teste.
        model_path: Caminho para o modelo salvo (formato JSON).
        label_names: Lista com os nomes das classes (opcional).
        report_path: Caminho para salvar o relatório de teste.
        """

        model_info = carregar_json(model_path) # Carrega os parâmetros do modelo previamente treinado a partir de um arquivo JSON

        # Inicializa os atributos do modelo com os dados carregados
        self.numberOfPerceptrons = model_info["numberOfPerceptrons"] # Número de neurônios na camada de saída
        self.numberOfInputs = model_info["numberOfInputs"] # Número de entradas (features)
        self.numberOfOutputs = model_info["numberOfOutputs"] # Número de saídas (classes, g eralmente igual aos perceptrons)
        self.bias = model_info["bias"] # Valor do bias
        self.weights = np.array(model_info["weights"]) # Vetor de pesos como numpy array

        # Converte os dados de teste para um array numpy
        test_data = np.array(test_data)

        # Inicializazção variável para cálculo das métricas 
        total_loss = 0 # Soma da perda cross-entropy
        total_mse = 0 # Soma dos erros quadráticos médios (MSE)
        correct_predictions = 0 # Contador de acertos
        predictions = [] # Lista para armazenar predições detalhadas

        # Loop para testar cada amostra do conjunto de teste
        for row in test_data:
            # Separa a entrada (x) da sáida esperada (y_true)
            *x, y_true = row[: -self.numberOfPerceptrons], row[-self.numberOfPerceptrons :]
            x = np.array(x, dtype=float) # Converte x para float
            y_true = np.array(y_true, dtype=int) # Converte y_true para inteiro

            # Executa o forward pass para obter a predição
            y_pred = self.forward(x)

            # Obtém a classe prevista e a classe verdadiera (posição do maior valor)
            y_pred_class = int(np.argmax(y_pred))
            y_true_class = int(np.argmax(y_true))

            # Calcula a perda e o erro quadrático para a amostra
            loss = self.cross_entropy(y_true, y_pred)
            mse = np.mean((y_true - y_pred) ** 2)

            # Acumula as métricas
            total_loss += loss
            total_mse += mse
            correct_predictions += int(y_pred_class == y_true_class)

            # Armazena os detalhes da predição atual
            predictions.append({
                "true_class": y_true_class,
                "predicted_class": y_pred_class,
                "probabilidades": y_pred.tolist(), # Probabilidade de cada classe como lista
            })

        # Calcula as médias das métricas
        avg_loss = total_loss / len(test_data)
        avg_mse = total_mse / len(test_data)
        accuracy = correct_predictions / len(test_data)

        # Monta o relatório final com as métricas e detalhes
        report = {
            "avg_cross_entropy_loss": round(avg_loss, 6), # Perda média cross-entropy
            "avg_mse_loss": round(avg_mse, 6), # MSE médio
            "accuracy": round(accuracy, 6), # Acurácia do modelo
            "total_samples": len(test_data), # total de amostras testadas
            "predictions": predictions, # Lista com predições individuais
        }

        garante_pasta(os.path.dirname(report_path))
        report_path += "test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Relatório de teste salvo em {report_path}")

    def main(self, file_path: str):
        # Lê o conteúdo bruto do arquivo (formato texto) contendo os dados do Wine.data
        raw_data = open_data(file_path)

        # Converte o texto bruto em uma matriz NumPy onde cada linha representa uma amostra
        data = self.from_raw_to_matrix(raw_data)

        # Normaliza os dados para o intervalo [-1, 1], o que ajuda no treinamento da rede
        data = self.normalize_between_minus1_1(data)

        # Aplica one-hot encoding nos rótulos (última coluna), transformando classes inteiras em vetorse binários
        data, _ = self.one_hotEncoder(data)

        # Embaralha os dados
        data = self.shuffle(data)

        # Separa os dados em treino, validação e teste
        train, val, test = self.divid_data(data=data)

        # Chama o experimento 1
        self.experiment1_testing_weightRange(train_data=train, val_data=val,
                                             test_data=test)
        
        # Chama o experimento 2
        self.experiment2_testing_learning_rate(train_data=train, val_data=val,
                                               test_data=test)

    def train(self, train, val, learning_rate=0.01, epochs=100, path="./Reports/"):
        train = np.array(train)
        training_log = []
        validation_log = []

        best_val_loss = float("inf")
        best_model_weights = None
        best_epoch = -1

        for epoch in range(epochs):
            total_loss = 0
            total_mse = 0
            for row in train:
                *x, y_true = row[: -self.numberOfPerceptrons], row[-self.numberOfPerceptrons :]
                x = np.array(x, dtype=float)
                y_true = np.array(y_true, dtype=int)

                y_pred = self.forward(x)
                loss = self.cross_entropy(y_true, y_pred)
                mse = np.mean((y_true - y_pred) ** 2)
                total_loss += loss
                total_mse += mse

                grad = self.softmax_cross_entropy_gradient(y_true, y_pred)
                x_bias = np.append(x, self.bias)
                for i in range(self.numberOfPerceptrons):
                    idx_start = i * (self.numberOfInputs + 1)
                    idx_end = idx_start + self.numberOfInputs + 1
                    self.weights[idx_start:idx_end] -= learning_rate * grad[i] * x_bias

            avg_loss = total_loss / len(train)
            avg_mse = total_mse / len(train)
            val_metrics = self.valid(val)

            if val_metrics["cross_entropy_loss"] < best_val_loss:
                best_val_loss = val_metrics["cross_entropy_loss"]
                best_model_weights = self.weights.copy()
                best_epoch = epoch

            training_log.append({
                "epoch": epoch + 1,
                "cross_entropy_loss": round(avg_loss, 6),
                "mse_loss": round(avg_mse, 6),
                "total_loss": round(total_loss, 6),
            })

            validation_log.append({
                "epoch": epoch + 1,
                **val_metrics
            })

            print(f"Época {epoch+1}/{epochs} - Entropia: {avg_loss:.4f} - MSE: {avg_mse:.4f}")

        # Salva o modelo final após todas as épocas
        self.save_model(f"{path}/model.json")

        # Salva o modelo com melhor validação
        self.weights = best_model_weights
        self.save_model(f"{path}/best_model_epoch_{best_epoch+1}.json")

        # Salva logs de treino e validação
        self.log_json(training_log, f"{path}/train_log.json")
        self.log_json(validation_log, f"{path}/validation_log.json")

    def experiment1_testing_weightRange(
        self, train_data, val_data, test_data, report_path: str = "./Reports/"
    ):
        weights_vector = []
        weights_tests = []
        fix_learning_rate = 0.1
        a = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            1.0,
        ]

        total_weights = self.numberOfPerceptrons * (self.numberOfInputs + 1)
        for j in a:
            for _ in range(total_weights):  # <=== Apenas esta linha foi corrigida
                weights_tests.append(j)
            weights_vector.append(weights_tests.copy())
            weights_tests = []

        test_path = report_path + "test_1/"
        garante_pasta(test_path)

        for weights in weights_vector:
            for i in range(10):
                nome = str(weights[0]).replace(".", "_")
                path = test_path + f"weights_{nome}/run_{i}/"
                garante_pasta(path)
                self.weights = []
                self.weights = np.array(weights)
                self.train(
                    train=train_data,
                    val=val_data,
                    learning_rate=fix_learning_rate,
                    path=path,
                )
                self.test(
                    test_data=test_data,
                    model_path=f"{path}model.json",
                    report_path=f"{path}",
                )

        # Teste com pesos aleatórios
        w = np.random.rand(total_weights)
        for i in range(10):
            self.weights = []
            self.weights = w.copy()
            random_path = test_path + f"weights_random/run_{i}/"
            garante_pasta(random_path)
            self.train(
                train=train_data,
                val=val_data,
                learning_rate=fix_learning_rate,
                path=random_path,)
            self.test(
                test_data=test_data,
                model_path=f"{random_path}model.json",
                report_path=f"{random_path}",
            )

    def experiment2_testing_learning_rate(
        self, train_data, val_data, test_data, report_path: str = "./Reports/"
    ):
        learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fix_weights = [0.5] * (self.numberOfPerceptrons * (self.numberOfInputs + 1))

        test_path = report_path + "test_2/"
        garante_pasta(test_path)

        for lr in learning_rates:
            for i in range(10):
                nome = str(lr).replace(".", "_")
                path = test_path + f"lr_{nome}/run_{i}/"
                garante_pasta(path)

                self.weights = np.array(fix_weights.copy())  # Garante que os pesos são resetados para 0.5 a cada iteração

                self.train(
                    train=train_data,
                    val=val_data,
                    learning_rate=lr,
                    path=path,
                )
                self.test(
                    test_data=test_data,
                    model_path=f"{path}/model.json",
                    report_path=f"{path}",
                )

def main():
    pn = PerceptronNetwork(numberOfPerceptrons=3, numberOfInputs=13, numberOfOutputs=3)
    pn.main(file_path="./DataSet/wine.data")

if __name__ == "__main__":
    main()

