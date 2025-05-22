import os
import json
import numpy as np
import pandas as pd

def garante_pasta(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)

def carregar_json(path: str):
    with open(path, "r") as f:
        return json.load(f)

def open_data(path: str):
    with open(path, "r") as f:
        return f.read()

class PerceptronNetwork:
    def __init__(self, numberOfPerceptrons: int, numberOfInputs: int, numberOfOutputs: int, bias: int = 1, weights=[]) -> None:
        self.numberOfPerceptrons = numberOfPerceptrons
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.bias = bias
        if weights == []:
            self.weights = np.random.rand(self.numberOfPerceptrons * (self.numberOfInputs + 1))

    def from_raw_to_matrix(self, data: str) -> np.ndarray:
        lines = data.strip().split("\n")
        dataset = []
        for line in lines:
            if line.strip():
                parts = line.split(',')
                cls = int(parts[0])  # classe é o primeiro valor
                features = [float(x) for x in parts[1:]]
                features.append(cls)
                dataset.append(features)
        return np.array(dataset, dtype=object)

    def normalize_between_minus1_1(self, data: np.ndarray) -> np.ndarray:
        features = data[:, :-1].astype(float)
        min_vals = features.min(axis=0)
        max_vals = features.max(axis=0)
        norm_features = 2 * (features - min_vals) / (max_vals - min_vals) - 1
        data[:, :-1] = norm_features
        return data

    def one_hotEncoder(self, data: np.ndarray):
        labels = data[:, -1]
        unique_cats = sorted(set(labels))
        cat_to_index = {cat: idx for idx, cat in enumerate(unique_cats)}
        one_hot = np.zeros((len(labels), len(unique_cats)), dtype=int)
        for i, label in enumerate(labels):
            one_hot[i, cat_to_index[label]] = 1
        return np.hstack((data[:, :-1], one_hot)), unique_cats

    def shuffle(self, data):
        np.random.shuffle(data)
        return data

    def divid_data(self, perTrain=0.7, perTest=0.15, perVal=0.15, data=[]):
        if perVal == 0.0:
            perVal = 1 - (perTrain + perTest)
        data_size = len(data)
        test_size = int(data_size * perTest)
        val_size = int(data_size * perVal)
        train_size = data_size - test_size - val_size
        test = data[:test_size]
        val = data[test_size:test_size + val_size]
        train = data[test_size + val_size:]
        return train, val, test

    def softMax(self, z: np.ndarray) -> np.ndarray:
        z = z.astype(np.float64)
        z_stable = z - np.max(z)
        exp_z = np.exp(z_stable)
        return exp_z / np.sum(exp_z)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.append(x, self.bias)
        weights_matrix = self.weights.reshape(self.numberOfPerceptrons, self.numberOfInputs + 1)
        logits = np.dot(weights_matrix, x)
        return self.softMax(logits)

    def cross_entropy(self, y_true, y_pred) -> float:
        return -np.sum(y_true * np.log(y_pred + 1e-15))

    def softmax_cross_entropy_gradient(self, y_true, y_pred) -> np.ndarray:
        return y_pred - y_true

    def valid(self, val_data):
        val_data = np.array(val_data)
        total_loss = 0
        total_mse = 0
        for row in val_data:
            *x, y_true = row[: -self.numberOfPerceptrons], row[-self.numberOfPerceptrons :]
            x = np.array(x, dtype=float)
            y_true = np.array(y_true, dtype=int)
            y_pred = self.forward(x)
            total_loss += self.cross_entropy(y_true, y_pred)
            total_mse += np.mean((y_true - y_pred) ** 2)
        return {
            "cross_entropy_loss": round(total_loss / len(val_data), 6),
            "mse_loss": round(total_mse / len(val_data), 6),
            "total_loss": round(total_loss, 6)
        }

    def save_model(self, file_path: str):
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

        model_info = carregar_json(model_path)
        self.numberOfPerceptrons = model_info["numberOfPerceptrons"]
        self.numberOfInputs = model_info["numberOfInputs"]
        self.numberOfOutputs = model_info["numberOfOutputs"]
        self.bias = model_info["bias"]
        self.weights = np.array(model_info["weights"])

        test_data = np.array(test_data)
        total_loss = 0
        total_mse = 0
        correct_predictions = 0
        predictions = []

        for row in test_data:
            *x, y_true = row[: -self.numberOfPerceptrons], row[-self.numberOfPerceptrons :]
            x = np.array(x, dtype=float)
            y_true = np.array(y_true, dtype=int)

            y_pred = self.forward(x)
            y_pred_class = int(np.argmax(y_pred))
            y_true_class = int(np.argmax(y_true))

            loss = self.cross_entropy(y_true, y_pred)
            mse = np.mean((y_true - y_pred) ** 2)

            total_loss += loss
            total_mse += mse
            correct_predictions += int(y_pred_class == y_true_class)

            predictions.append({
                "true_class": y_true_class,
                "predicted_class": y_pred_class,
                "probabilidades": y_pred.tolist(),
            })

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

        garante_pasta(os.path.dirname(report_path))
        report_path += "test_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)

        print(f"Relatório de teste salvo em {report_path}")

    def main(self, file_path: str):
        raw_data = open_data(file_path)
        data = self.from_raw_to_matrix(raw_data)
        data = self.normalize_between_minus1_1(data)
        data, _ = self.one_hotEncoder(data)
        data = self.shuffle(data)
        train, val, test = self.divid_data(data=data)
        # self.train(train, val)
        # self.test(test_data=test, model_path='./Reports/model.json')
        self.experiment1_testing_weightRange(train_data=train, val_data=val,
                                             test_data=test)
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

