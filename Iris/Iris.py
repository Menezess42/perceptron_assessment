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


import json
import os

# IMPORT
import numpy as np
import pandas as pd


def garante_pasta(path: str) -> None:
    """
    path: str - Caminho da pasta
    """
    # Verifica se a pasta existe
    if not os.path.exists(path):
        # Se não existe então cria
        os.makedirs(path)


def carregar_json(path: str):
    with open(path, "r") as f:
        return json.load(f)


def open_data(path: str):
    with open(path, "r") as f:
        return f.read()


class PerceptronNetwork:
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
            self.weights = np.random.rand(
                self.numberOfPerceptrons * (self.numberOfInputs + 1)
            )

    def save_model(self, file_path: str):
        """
        salva o modelo em um arquvio JSON.
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
        split1 = data.split("\n")
        a = []

        for item in split1:
            if not item.strip():
                continue
            parts = item.split(",")
            *floats, last = parts
            floats = [float(x) for x in floats]
            floats.append(last)
            a.append(floats)

        return a

    def divid_data(
        self, perTrain: float = 0.7, perTest: float = 0.15, perVal: float = 0.0, data=[]
    ) -> tuple:

        if perVal == 0.0:
            perVal = 1 - (perTrain + perTest)
        data_size = len(data)
        test_size = int(data_size * perTest)
        val_size = int(data_size * perVal)
        train_size = int(data_size * perTrain)
        test = data[:test_size]
        val = data[test_size : test_size + val_size]
        train = data[test_size + val_size :]
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
        z = z.astype(np.float64)
        z_stable = z - np.max(z)
        exp_z = np.exp(z_stable)
        softmax = exp_z / np.sum(exp_z)
        return softmax

    def train(
        self, train, val, learning_rate=0.01, epochs=100, path: str = "./Reports/"
    ):
        """
        Treina a rede usando gradiente descendente.
        train: dados de treinamento (entrada + saída one-hot)
        """
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
                *x, y_true = (
                    row[: -self.numberOfPerceptrons],
                    row[-self.numberOfPerceptrons :],
                )
                x = np.array(x, dtype=float)
                y_true = np.array(y_true, dtype=int)

                y_pred = self.forward(x)

                loss = self.cross_entropy(y_true, y_pred)
                total_loss += loss

                mse = np.mean((y_true - y_pred) ** 2)
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

            training_log.append(
                {
                    "epoch": epoch + 1,
                    "cross_entropy_loss": round(avg_loss, 6),
                    "mse_loss": round(avg_mse, 6),
                    "total_loss": round(total_loss, 6),
                }
            )
            validation_log.append(
                {
                    "epoch": epoch + 1,
                    "cross_entropy_loss": val_metrics["cross_entropy_loss"],
                    "mse_loss": val_metrics["mse_loss"],
                    "total_loss": val_metrics["total_loss"],
                }
            )

            print(
                f"Época {epoch+1}/{epochs} - Entropia: {avg_loss:.4f} - MSE: {avg_mse:.4f}"
            )

        self.log_train_report(training_log, path)
        self.log_validation_report(validation_log, path)

        # Salva o modelo após o treinamento
        # save_path = "./Reports/model.json"
        save_path = path + "model.json"
        self.save_model(save_path)
        self.weights = best_model_weights
        self.save_model(f"{path}best_model_epoch_{best_epoch+1}.json")

    def cross_entropy(self, y_true, y_pred) -> float:
        return -np.sum(y_true * np.log(y_pred + 1e-15))

    def softmax_cross_entropy_gradient(self, y_true, y_pred) -> np.ndarray:
        return y_pred - y_true

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
        self.numberOfPerceptrons = model_info["numberOfPerceptrons"]
        self.numberOfInputs = model_info["numberOfInputs"]
        self.numberOfOutputs = model_info["numberOfOutputs"]

        test_data = np.array(test_data)
        total_loss = 0
        total_mse = 0
        correct_predictions = 0
        predictions = []

        for row in test_data:
            *x, y_true = (
                row[: -self.numberOfPerceptrons],
                row[-self.numberOfPerceptrons :],
            )
            x = np.array(x, dtype=float)
            y_true = np.array(y_true, dtype=int)

            y_pred = self.forward(x)
            y_pred_class = np.argmax(y_pred)
            y_true_class = np.argmax(y_true)

            loss = self.cross_entropy(y_true, y_pred)
            mse = np.mean((y_true - y_pred) ** 2)

            total_loss += loss
            total_mse += mse

            if y_pred_class == y_true_class:
                correct_predictions += 1

            predictions.append(
                {
                    "true_class": int(y_true_class),
                    "predicted_class": int(y_pred_class),
                    "probabilidades": y_pred.tolist(),
                }
            )

        avg_loss = total_loss / len(test_data)
        avg_mse = total_mse / len(test_data)
        accuracy = correct_predictions / len(test_data)

        report = {
            "avg_cross_entropy_loss": round(avg_loss, 6),
            "avg_mse_loss": round(avg_mse, 6),
            "accuracy": round(accuracy, 6),
            "total_samples": len(test_data),
            "predictions": predictions,
        }

        # garante_pasta(os.path.dirname(report_path))
        report_path += "test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Relatório de teste salvo em {report_path}")

    def valid(self, val_data):
        val_data = np.array(val_data)
        total_loss = 0
        total_mse = 0
        for row in val_data:
            *x, y_true = (
                row[: -self.numberOfPerceptrons],
                row[-self.numberOfPerceptrons :],
            )
            x = np.array(x, dtype=float)
            y_true = np.array(y_true, dtype=int)
            y_pred = self.forward(x)

            loss = self.cross_entropy(y_true, y_pred)
            mse = np.mean((y_true - y_pred) ** 2)

            total_loss += loss
            total_mse += mse

        avg_loss = total_loss / len(val_data)
        avg_mse = total_mse / len(val_data)

        return {
            "cross_entropy_loss": round(avg_loss, 6),
            "mse_loss": round(avg_mse, 6),
            "total_loss": round(total_loss, 6),
        }

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
        """
        raw_data = open_data(file_path)
        data = self.from_raw_to_matrix(raw_data)
        hoted_data, unique_cats = self.one_hotEncoder(data)
        shuffled_data = self.shuffle(hoted_data)
        train, val, test = self.divid_data(
            perTrain=0.7, perTest=0.15, data=shuffled_data
        )
        self.experiment1_testing_weightRange(train_data=train, val_data=val, test_data=test)
        self.experiment2_testing_learning_rate(train_data=train, val_data=val, test_data=test)

    def experiment1_testing_weightRange(
        self, train_data, val_data, test_data, report_path: str = "./Reports/"
    ):
        weights_vector = []
        weights_tests = []
        fix_learning_rate = 0.1  # <== Defina aqui o valor que quiser
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
        ]  # removido o 0.5 duplicado

        for j in a:
            for _ in range(self.numberOfInputs * self.numberOfInputs + 1):
                weights_tests.append(j)
            weights_vector.append(weights_tests.copy())  # garante cópia independente
            weights_tests = []

        test_path = report_path + "test_1/"
        garante_pasta(test_path)

        for weights in weights_vector:
            nome = str(weights[0]).replace(
                ".", "_"
            )  # evita problema com ponto no nome da pasta
            path = test_path + f"weights_{nome}/"
            garante_pasta(path)
            self.weights = weights
            self.train(
                train=train_data,
                val=val_data,
                learning_rate=fix_learning_rate,
                path=path + "train_report.json",
            )
            self.test(
                test_data=test_data,
                model_path=f"{path}/model.path",
                report_path=f"{path}/test_report.json",
            )
        random_path = test_path + "weights_random/"
        garante_pasta(random_path)
        self.weights = np.random.rand(
            self.numberOfPerceptrons * (self.numberOfInputs + 1)
        )
        self.train(
            train=train_data,
            val=val_data,
            learning_rate=fix_learning_rate,
            path=random_path + "train_report.json",
        )

        self.test(
            test_data=test_data,
            model_path=f"{random_path}/model.path",
            report_path=f"{random_path}/test_report.json",
        )

    def experiment2_testing_learning_rate(
        self, train_data, val_data, test_data, report_path: str = "./Reports/"
    ):
        fix_learning_rates = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        fix_weights = [0.5] * (self.numberOfInputs * self.numberOfInputs + 1)

        test_path = report_path + "test_2/"
        garante_pasta(test_path)

        for lr in fix_learning_rates:
            nome = str(lr).replace(".", "_")
            path = test_path + f"lr_{nome}/"
            garante_pasta(path)

            self.weights = (
                fix_weights.copy()
            )  # Garante que os pesos são resetados para 0.5 a cada iteração

            self.train(
                train=train_data,
                val=val_data,
                learning_rate=lr,
                path=path + "train_report.json",
            )
            self.test(
                test_data=test_data,
                model_path=f"{path}/model.path",
                report_path=f"{path}/test_report.json",
            )


def main():
    pn = PerceptronNetwork(numberOfPerceptrons=3, numberOfInputs=4, numberOfOutputs=3)
    pn.main(file_path="./DataSet/iris.data")


if __name__ == "__main__":
    main()
