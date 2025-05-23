def instruction():
    """
    # comments on How to make the experiment
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
    """
# IMPORT
import numpy as np
import pandas as pd
import json
import os

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
    Classe principal. Reponsável por criar a rede perseptron e realizar os testes no dataSet.
    """
    def __init__(
        self,
        numberOfPerceptrons: int,
        numberOfInputs: int,
        numberOfOutputs: int,
        bias: int = 1,
        weights=[],
    ) -> None:
        """
        numberOfPerceptrons: Numero de perceptrons
        numberOfInputs: numero de inputs
        numberOfOutputs: Numero de outputs
        bias: bias, default=1 weights: Vetor de pesos dos inputs; Se deixado vazio é
        preenchido por valores vazios
        """

        self.numberOfPerceptrons = numberOfPerceptrons
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.bias = bias

        if weights == []:
            # Se weights não é definido ao chamar a classe, então os pesos são definidos aleatóriamente.
            self.weights = np.random.rand(
                self.numberOfPerceptrons * (self.numberOfInputs + 1)
            )

    def save_model(self, file_path: str):
        """
        salva o modelo em um arquvio JSON.
        file_path: caminho onde o modelo deve ser salvo.
        """
        model_data = {
            "numberOfPerceptrons": self.numberOfPerceptrons,
            "numberOfInputs": self.numberOfInputs,
            "numberOfOutputs": self.numberOfOutputs,
            "bias": self.bias,
            "weights": self.weights.tolist(),
        }

        with open(file_path, "w") as f:
            json.dump(model_data, f, indent=4)

        print(f"Modelo salvo em {file_path}")

    def shuffle(self, data):
        """
        Função responsável por embaralhar os dados utilizando a função shuffle do numpy
        """
        np.random.shuffle(data) # Recebe os dados e embaralha eles
        return data

    def one_hotEncoder(self, data):
        """
        Função responsável por transformar classes categóricas (os rótulos dos dados) em classes binárias onde
        cada grupo de bits representa uma categoria das classes.
        data: dataset.
        """
        data = np.array(data, dtype=object) # função responsável por converter o data[array] para data[numpy array].
        labels = data[:, -1] # Separa os rótulos dos dados do resto do dataset.
        unique_cats = sorted(set(labels)) # Pega 1 de cada rótulo.
        cat_to_index = {cat: idx for idx, cat in enumerate(unique_cats)} # Cria um dicionário onde cada categoria(cat)
        # da lista unique_catas é associada a seu índice (idx) correspondente.

        one_hot = np.zeros((len(labels), len(unique_cats)), dtype=int) # Cria a matriz de binários correspondente a quantidade de rótulos únicos.
        for i, label in enumerate(labels): # para i=Contador, label=Rótulo
            col = cat_to_index[label] # Pega a coluna do cat_to_index
            one_hot[i, col] = 1 # Liga esta coluna, ou seja, [0 0 0] para [1 0 0]

        data = np.hstack((data[:, :-1], one_hot))# Remove a coluna original e concatena one-hot
        return data, unique_cats

    def from_raw_to_matrix(self, data: str) -> np.matrix:
        """
        data: DataSet vindo direto da leitura Iris.data
        Função responsável por ler o Iris.data, que contém os dados separados por virgulas
        e as linhas separados por \n e converte para dados em uma array.
        """
        split1 = data.split("\n") # Quebra o data(Uma grande string) no \n e cria um data com várias strings menóres onde cada
        # string menor é uma linha do dataset.

        a = []

        for item in split1: # para cada linha no split1
            if not item.strip():
                continue
            parts = item.split(",") # Pega as linhas e quebra elas pela separação dos dados, que neste caso é a virgula.
            *floats, last = parts # Floats==Recebe todos os dados do dataset menos a úlima coluna(last) que possui os rótulos.
            floats = [float(x) for x in floats] # Converte as strings contidas em floats para dados do tipo floats
            floats.append(last) # Junta tudo em uma array só, junta todos os floats e os rótulos
            a.append(floats) # Adiciona essa array na array principal.

        return a

    def divid_data(
        self, perTrain: float = 0.7, perTest: float = 0.15, perVal: float = 0.0, data=[]
    ) -> tuple:
        """
        perTrain (float): A porcentagem do dataSet que será reservado aos dados de treino.
        perTest (float): A porcentagem do dataSet que será reservado aos dados de teste.
        perVal (float): A porcentagem do dataSet que será reservada aos dados de validação.
        data (array): O dataSet.
        Retorna uma tupla contendo (train, val, test).
        """
        if perVal == 0.0:
            perVal = 1 - (perTrain + perTest) # Se a porcentagem da validação não for definida, ela é calculada
            # sendo o respo após a porcentagem de treino e teste.
        data_size = len(data) # pega o tamanho do dataset.
        test_size = int(data_size * perTest) # Define o tamanho(quantidade) dos dados de teste.
        val_size = int(data_size * perVal) # Define o tamanho(quantidade) dos dados de validação.
        train_size = int(data_size * perTrain) # Define o tamanho(quantidade) dos dados de treino
        test = data[:test_size] # Separa os dados de teste.
        val = data[test_size : test_size + val_size] # Separa os dados de validação.
        train = data[test_size + val_size :] # Separa os dados de treino.
        return train, val, test

    def softMax(self, z: np.ndarray) -> np.ndarray:
        """
        Aplica a função softmax sobre o vetor de saídas
        z dos perceptrons.
        Z: vetor numpy com as ativações dos perceptrons
        (shape: [numberOfPerceptrons])
        Retorna: Vetor com as probabilidades normalizadas
        (shape: [numberOfPerceptrons])
        """
        z = z.astype(np.float64)  # Converte os valores de z para float64 para garantir precisão numérica
        z_stable = z - np.max(z)  # Subtrai o maior valor de z de todos os elementos para evitar overflow numérico
        exp_z = np.exp(z_stable)  # Aplica a função exponencial elemento a elemento no vetor estabilizado
        softmax = exp_z / np.sum(exp_z)  # Normaliza os valores exponenciais dividindo pela soma total, obtendo probabilidades
        return softmax  # Retorna o vetor resultante com as probabilidades normalizadas

    def train(
        self, train, val, learning_rate=0.01, epochs=100, path: str = "./Reports/"
    ):
        """
        Treina a rede usando gradiente descendente.
        train: dados de treinamento (entrada + saída one-hot).
        val: Dados de validação.
        learning_rate (float): Define a taxa de aprendizado, responsável por dar os saltos na função que anda em direção ao mínimo global.
        epochs (int): Quantidades de épocas que o modelo vai ser treinado.
        path: Caminho para salvar os modelos.
        """
        train = np.array(train) # Converte os dados train array[array] em dados numpyArray[numpyArray].
        training_log = [] # Array para salvar os logs de treinamento.
        validation_log = [] # Array para salvar os logs de validação.

        best_val_loss = float("inf") # Melhor valor de validação; Utilizado para salvar o melhor modelo definido pela função de validação.
        best_model_weights = None # Melhor pesos do melhor modelo definido pela função de validação.
        best_epoch = -1 # Melhor época definida pela função de validação.

        for epoch in range(epochs): # Para época em épocas
            total_loss = 0
            total_mse = 0

            for row in train: # Para cada coluna no dado de treino
                *x, y_true = (
                    row[: -self.numberOfPerceptrons],  # Pega todos os dados menos as colunas de rótulo.
                    row[-self.numberOfPerceptrons :], # Pega as colunas de rótulos.
                )
                x = np.array(x, dtype=float) # Converte o array de dados para numpyArray de dados.
                y_true = np.array(y_true, dtype=int) # Converte as colunas de rótulo de array para numpyArray.

                y_pred = self.forward(x) # Alimenta com x a função que passa os dados pela rede e recebe de retorno os rótulos preditos

                loss = self.cross_entropy(y_true, y_pred) # Recebe a a perda em relação aos rótulos verdadeiros e os preditos
                total_loss += loss # Soma a perda a perda total.

                mse = np.mean((y_true - y_pred) ** 2) # Calcula o erro quadrático médio.
                total_mse += mse # Adiciona o mse ao total mse.

                grad = self.softmax_cross_entropy_gradient(y_true, y_pred) # Calcula o gradiente descendente para softmax.

                x_bias = np.append(x, self.bias) # Adiciona a bias aos dados.

                for i in range(self.numberOfPerceptrons):  
                    # Itera sobre cada perceptron da camada

                    idx_start = i * (self.numberOfInputs + 1)  
                    # Calcula o índice inicial dos pesos correspondentes a esse perceptron (incluindo o bias)

                    idx_end = idx_start + self.numberOfInputs + 1  
                    # Calcula o índice final dos pesos desse perceptron

                    self.weights[idx_start:idx_end] -= learning_rate * grad[i] * x_bias  
                    # Atualiza os pesos desse perceptron com base no gradiente calculado (grad[i]),


            avg_loss = total_loss / len(train) # Calcula a perda média.
            avg_mse = total_mse / len(train) # calcula a média do erro quadrático médio.

            val_metrics = self.valid(val) # Passa o modelo dessa época pela função de validação e recebe de volta as métricas de validação.

            if val_metrics["cross_entropy_loss"] < best_val_loss: # Se os dados que chegaram da função de validação são melhores que os anteriormente 
                # armazenados, então ele atualiza a melhor perda, peso do modelo e a melhor época.
                best_val_loss = val_metrics["cross_entropy_loss"]
                best_model_weights = self.weights.copy()
                best_epoch = epoch

            training_log.append(
                {
                    "epoch": epoch + 1,
                    "cross_entropy_loss": round(avg_loss, 6),
                    "mse_loss": round(avg_mse, 6),
                    "total_loss": round(total_loss, 6),
                } # Cataloga as métricas após o treinamento desta época.
            )
            validation_log.append(
                {
                    "epoch": epoch + 1,
                    "cross_entropy_loss": val_metrics["cross_entropy_loss"],
                    "mse_loss": val_metrics["mse_loss"],
                    "total_loss": val_metrics["total_loss"],
                }
            ) # Cataloga as métricas após a validação deta época.

            print(
                f"Época {epoch+1}/{epochs} - Entropia: {avg_loss:.4f} - MSE: {avg_mse:.4f}"
            )

        self.log_train_report(training_log, path) # Salva em um arquivo .json os logs de treinamento
        self.log_validation_report(validation_log, path) # Salva em um arquivo .json os logs de validação 

        # Salva o modelo após o treinamento
        save_path = path + "model.json"
        self.save_model(save_path)
        
        # Salva o melhor modelo definido pela função de validação
        self.weights = best_model_weights
        self.save_model(f"{path}best_model_epoch_{best_epoch+1}.json")
    
    def cross_entropy(self, y_true, y_pred) -> float:
        """
        Calcula a função de perda cross-entropy (entropia cruzada)
        y_true: vetor one-hot com os valores reais das classes (ex: [0, 1, 0])
        y_pred: vetor com as probabilidades previstas pelo modelo (ex: [0.2, 0.7, 0.1])
        """
        return -np.sum(y_true * np.log(y_pred + 1e-15))  
        # Multiplica os valores reais pelas log-probabilidades previstas
        # Soma todos os resultados e inverte o sinal para obter a perda
        # O valor 1e-15 evita log(0), que causaria erro de log 0

    def softmax_cross_entropy_gradient(self, y_true, y_pred) -> np.ndarray:
        """
        Calcula o gradiente da função de perda cross-entropy com softmax
        y_true: vetor one-hot com a classe correta
        y_pred: vetor com as probabilidades previstas (após softmax)
        """
        return y_pred - y_true  
        # O gradiente da cross-entropy combinada com softmax é simplesmente a diferença entre a saída prevista e a saída real

    def log_train_report(
        self, report_data: list[dict], file_path: str = "./Reports/train_log.json"
    ) -> None:
        """
        Salva o log de treinamento em JSON.
        report_data: Lista de dicionários contendo métricas por época.
        file_path: Caminho para salvar o arquivo JSON.
        """
        # garante_pasta(os.path.dirname(file_path))
        file_path += "validation_log.json"
        with open(file_path, "w") as f:
            json.dump(report_data, f, indent=4)

        print(f"Relatório de treinamento salvo em {file_path}")

    def log_validation_report(
        self, report_data: list[dict], file_path: str = "./Reports/validation_log.json"
    ) -> None:
        # garante_pasta(os.path.dirname(file_path))
        file_path += "validation_log.json"
        with open(file_path, "w") as f:
            json.dump(report_data, f, indent=4)

        print(f"Relatório de validação salvo em {file_path}")

    def test(
        self,
        test_data,
        model_path: str,
        label_names=None,
        report_path="./Reports/test_report.json",
    ):
        """
        Avalia o modelo carregado com os dados de teste.

        test_data: Lista ou array com as amostrar de teste.
        model_path: Caminho para o modelo salvo (formato JSON).
        label_names: List com os nomes das classes, usado na matriz de
        confusão.
        Report_path: Caminho para salvar o relatório de teste.
        """

        # Carrega o modelo salvo
        model_info = carregar_json(model_path)
        self.numberOfPerceptrons = model_info["numberOfPerceptrons"] # Passa para a variável global da classe a quantidade de perceptrons.
        self.numberOfInputs = model_info["numberOfInputs"] # Passa para a variável global da classe a quantidade de inputs.
        self.numberOfOutputs = model_info["numberOfOutputs"] # Passa para a variável global da classe a quantidade de Outputs.

        test_data = np.array(test_data) # Converte test_data de array para numpyArray
        total_loss = 0 
        total_mse = 0
        correct_predictions = 0
        predictions = []

        for row in test_data: # Para cada linha no dataSet de test
            *x, y_true = (
                row[: -self.numberOfPerceptrons], # Coleta todos os dados da linha menos as colunas responsáveis pelos rótulos.
                row[-self.numberOfPerceptrons :], # Coleta as colunas responsáveis pelos rótulos.
            )

            x = np.array(x, dtype=float) # Converte os dados x de array para numpyArray.
            y_true = np.array(y_true, dtype=int) # Converte os dodos responsáveis pelos rótulos de array para numpyArray.

            y_pred = self.forward(x) # Alimenta com x a função que passa os dados pela rede e recebe de retorno os rótulos preditos.
            y_pred_class = np.argmax(y_pred) # classe predita.
            y_true_class = np.argmax(y_true) # classe esperada.

            loss = self.cross_entropy(y_true, y_pred) # Entropia cruzada entre a classe predita e a esperada.
            mse = np.mean((y_true - y_pred) ** 2) # Erro quadrático médio em relação a classe predita pela classe esperada.

            total_loss += loss # Adiciona a perda a perda total.
            total_mse += mse # Adiciona o mse ao mse total.

            if y_pred_class == y_true_class:
                # Se a classe predita é igual a esperada
                correct_predictions += 1 # então acresce em 1 o contador de predições corretas.

            predictions.append(
                {
                    "true_class": int(y_true_class),
                    "predicted_class": int(y_pred_class),
                    "probabilidades": y_pred.tolist(),
                }
            ) # Cataloga as métricas de predição.

        avg_loss = total_loss / len(test_data) # Calcula a média de perda.
        avg_mse = total_mse / len(test_data) # Calcula a média de mse.
        accuracy = correct_predictions / len(test_data) # Cacula a acurácia, com a variável de qtde de predições corretas
        # dividida pela quantidade de linhas no dataSet de teste.

        report = {
            "avg_cross_entropy_loss": round(avg_loss, 6),
            "avg_mse_loss": round(avg_mse, 6),
            "accuracy": round(accuracy, 6),
            "total_samples": len(test_data),
            "predictions": predictions,
        } # Cria um log da capacidade do modelo no teste.

        # Salva a capacidade do modelo no teste em um arquivo json.
        report_path += "test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Relatório de teste salvo em {report_path}")

    def valid(self, val_data):
        """
        Função de validação. Responsável por validar o modelo por cada época do treinamento.
        Não executa mudanças no modelo, apenas valida o modelo.
        val_data: Dados de validação.
        """
        val_data = np.array(val_data) # Converte val_data de array para numpyArray.
        total_loss = 0 
        total_mse = 0

        for row in val_data: # Para cada linha em val_data
            *x, y_true = (
                row[: -self.numberOfPerceptrons], # Recebe toda as colunas de dados.
                row[-self.numberOfPerceptrons :], # Recebe as colunas responsáveis pelos rótulos.
            )
            x = np.array(x, dtype=float) # Converte data x do tipo array para o tipo numpyArray.
            y_true = np.array(y_true, dtype=int) # Converte o data y_true do tipo array para o tipo numpyArray.
            y_pred = self.forward(x)  # Alimenta com x a função que passa os dados pela rede e recebe de retorno os rótulos preditos 

            loss = self.cross_entropy(y_true, y_pred) # Calcula a entropia cruzada entre a classe predita e a esperada.
            mse = np.mean((y_true - y_pred) ** 2) # Calcula o erro quadrático médio entre a classe predita e a esperada.

            total_loss += loss # Adiciona a perda ao total.
            total_mse += mse # Adiciona o mse ao total.

        avg_loss = total_loss / len(val_data) # Calcula a perda média.
        avg_mse = total_mse / len(val_data) # Calcula o mse médio.

        return {
            "cross_entropy_loss": round(avg_loss, 6),
            "mse_loss": round(avg_mse, 6),
            "total_loss": round(total_loss, 6),
        } # Retorna as devidas métricas.

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Calcula a saída da rede (logists)+ aplica softmax.
        x: vetor de entrada (shape: [numberOfInputs])
        retorna: vetor de probabilidades (shape: [numberOfPerceptrons])
        """
        # Adiciona o bias como entrada extra
        x = np.append(x, self.bias)

        # Grante que os pesos estão reshaped corretamente
        weights_matrix = self.weights.reshape(
            self.numberOfPerceptrons, self.numberOfInputs + 1
        )

        # Produto escalar de cadad neurônio com os inputs
        logits = np.dot(weights_matrix, x)

        # Softmax para converter logits em probabilidades
        returns = self.softMax(logits)
        return returns

    def main(self, file_path: str):
        """
        file_path: Caminho dos dados
        Função principal. Executa os testes.
        """
        raw_data = open_data(file_path) # Le os dados do Iris.data
        data = self.from_raw_to_matrix(raw_data) # Converte os dados de uma grande string para uma Matriz.
        hoted_data, unique_cats = self.one_hotEncoder(data) # Converte os rótulos categóricos em categórias binárias.
        shuffled_data = self.shuffle(hoted_data) # Embaralha os dados.
        train, val, test = self.divid_data( # Separa os dados em dados de treino, validação e teste
            perTrain=0.7, perTest=0.15, data=shuffled_data
        )

        self.experiment1_testing_weightRange(train_data=train, val_data=val, test_data=test) # Realiza o experimento 1: Pesos variados.
        self.experiment2_testing_learning_rate(train_data=train, val_data=val, test_data=test) # Realiza o experimento 2: Taxas de aprendizado variadas.

    def experiment1_testing_weightRange(
        self, train_data, val_data, test_data, report_path: str = "./Reports/"
    ):
        """
        train_data: Dados de treinamento.
        val_data: Dados de validação. 
        test_data: Dados de teste.
        report_path: Pasta onde os relatórios devem ser gravados.
        Essa função é responsável por rodar o experimento 1 que consiste em testar o modelo com uma série de pesos iniciais e depois
        com pesos iniciais aleátorios.
        """
        weights_vector = []
        weights_tests = []
        fix_learning_rate = 0.1 # Taxa de aprendizado fixa que não altera entre as iterações
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
        ] # Set de peso de inicialização para serem testados no modelo.

        total_weights = self.numberOfPerceptrons * (self.numberOfInputs + 1) # Quantidade de pesos que precisamos.

        for j in a: # Neste for estamos configurando o vetor de pesos para cada peso do experimento.
            for _ in range(total_weights):  # <=== Apenas esta linha foi corrigida.
                weights_tests.append(j)
            weights_vector.append(weights_tests.copy())
            weights_tests = []

        test_path = report_path + "test_1/"
        garante_pasta(test_path)

        for weights in weights_vector: # Para vetor de pesos na matriz de pesos
            for i in range(10): # Aqui executamos o experimento de rodar 10x para cada set de pesos.
                nome = str(weights[0]).replace(".", "_") # Configuramos os pesos em um formato para ser o nome dos arquivos.
                path = test_path + f"weights_{nome}/run_{i}/"
                garante_pasta(path)
                self.weights = [] # Limpamos os pesos da classe
                self.weights = np.array(weights) # Configuramos os pesos da classe para ser o vetor de pesos atual do experimento.
                self.train(
                    train=train_data,
                    val=val_data,
                    learning_rate=fix_learning_rate,
                    path=path,
                ) # Aqui treinamos o modelo
                self.test(
                    test_data=test_data,
                    model_path=f"{path}model.json",
                    report_path=f"{path}",
                ) # Aqui testamos o modelo

        # Teste com pesos aleatórios
        w = np.random.rand(total_weights)
        # Após termos testado o modelo com todos os sets de pesos definidos, fazemos um último teste que é
        # rodar mais 10x o modelo treinando e testando, mas desta vez com pesos definidos aleatóriamente.
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
        """
        train_data: Dados de treinamento.
        val_data: Dados de validação. 
        test_data: Dados de teste.
        report_path: Pasta onde os relatórios devem ser gravados.
        Essa função é responsável por rodar o experimento 2 que consiste em testar o modelo com uma série de taxas de aprendizado 
        """
        # Aqui definimos o vetor de taxas de aprendizados que iremos testar, assim como configurar o peso inicial que é fixo em 0.5
        # para todos os experimentos.
        learning_rates = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fix_weights = [0.5] * (self.numberOfPerceptrons * (self.numberOfInputs + 1))

        test_path = report_path + "test_2/"
        garante_pasta(test_path)

        for lr in learning_rates: # Para cada learning rate em learning rates
            for i in range(10): # Roda cada experimento de learning rate 10x.
                
                # Configuração para criar as pastas de cada experimento.
                nome = str(lr).replace(".", "_")
                path = test_path + f"lr_{nome}/run_{i}/"
                garante_pasta(path)

                self.weights = np.array(fix_weights.copy())  # Garante que os pesos são resetados para 0.5 a cada iteração

                self.train(
                    train=train_data,
                    val=val_data,
                    learning_rate=lr,
                    path=path,
                ) # Treina o modelo.
                self.test(
                    test_data=test_data,
                    model_path=f"{path}/model.json",
                    report_path=f"{path}",
                ) # Testa o modelo.


def main():
    pn = PerceptronNetwork(numberOfPerceptrons=3, numberOfInputs=4, numberOfOutputs=3)
    pn.main(file_path="./DataSet/iris.data")


if __name__ == "__main__":
    main()
