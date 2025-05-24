import json
import os
import re
from pathlib import Path

import pandas as pd


class Analyzer_test_1:
    """
    Classe responsável por criar o dataframe para o experimento 1.
    Ou seja, classe responsável por ler os .json de test_1 e transformar em dataFrame.
    """
    def __init__(self, base_dir="./Reports/test_1/"):
        """
        Carrega o caminho onde os relatórios .json então.
        """
        self.base_dir = Path(base_dir)
        self.data = []

    def load_all(self):
        """
        Carrega todos os dados.
        """
        for weight_dir in sorted(self.base_dir.glob("weights_*")):
            # Itera sobre todos os diretórios que começam com "weights_" (ex: weights_0_1)
            # Usa sorted para grantir a ordem consistente na leitura
            weight_key = weight_dir.name.replace("weights_", "")
            # Extrai a chave do conjunto de pesos, removendo o prefixo "weights_"
            # por exemplo, "weights_0_1" vira "0_1"

            for run_dir in sorted(weight_dir.glob("run_*")):
                # Itera sobre os diretórios de execução (run_*) dentro de cada diretório de pesos
                # Cada run representa uma execução ou experimento diferente
                run_index = int(run_dir.name.replace("run_", ""))
                # Extrai o índice da run (ex: "run_2" vira 2)
                record = {
                    "weight_init": weight_key, # Salva o chave de inicialização dos pesos
                    "run": run_index, # Salva o número da execução
                }

                best_model_file = next(run_dir.glob("best_model_epoch_*.json"), None)
                # Procura o arquio JSON com o melhor modelo salvo (com nome tipo: best_model_epoch_5.json)
                # Se não encontrar, best_model_file fica como None
                if best_model_file:
                    # Se encontrou o arquivo com o melhor modelo
                    match = re.search(
                        r"best_model_epoch_(\d+)\.json", best_model_file.name
                    ) # Extrai o número da época (epoch) do nome do arquivo usano regex
                    if match:
                        record["best_model_epoch"] = int(match.group(1))
                        # Salva a época em que o melhor modelo foi salvo.
                    with open(best_model_file, "r") as f:
                        record["best_model"] = json.load(f)
                        # Abre o arquivo JSON e carrega os dados do melhor modelo para dicionário 'record'

                model_file = run_dir / "model.json"
                # Deine o caminho para o arquivo com o modelo final da run
                if model_file.exists():
                    # Verifica se o arquivo do modelo final existe
                    with open(model_file, "r") as f:
                        record["final_model"] = json.load(f)
                        # Carrega o modelo final da run e armazena no dicionário 'record'

                test_report_file = run_dir / "test_report.json"
                # Define o caminho para o relatório de teste da run
                if test_report_file.exists():
                    # Verifica se o arquivo de teste existe
                    with open(test_report_file, "r") as f:
                        test_data = json.load(f)
                        # Carrega os dados do relatório de teste
                        record.update(
                            {
                                "accuracy": test_data.get("accuracy"),
                                "avg_cross_entropy_loss": test_data.get(
                                    "avg_cross_entropy_loss"
                                ),
                                "total_samples": test_data.get("total_samples"),
                                "predictions": test_data.get("predictions"),
                            }
                        )

                validation_file = run_dir / "validation_log.json"
                # Define o caminho para o log de validação da run
                if validation_file.exists():
                    # verifica se o arquivo de validação existe
                    with open(validation_file, "r") as f:
                        record["validation_log"] = json.load(f)
                        # Carrega o log de validação e armazena no dicionário 'record'

                self.data.append(record)

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def save_dataframe(self, path=None):
        df = self.to_dataframe()
        path = path or self.base_dir / "compiled_data_test_1.pkl"
        df.to_pickle(path)

    def load_dataframe(self, path=None):
        path = path or self.base_dir / "compiled_data.pkl"
        return pd.read_pickle(path)


class Analyzer_test_2:
    """
    Classe responsável por criar o dataFrame para o experimento 2.
    Ou seja, classe responsável por ler os .json de test_2 e transformar em dataFrame.
    """
    def __init__(self, base_dir="./Reports/test_2/"):
        """
        Carrega o caminho onde os relatórios .json estão.
        """
        self.base_dir = Path(base_dir)
        self.data = []

    def load_all(self):
        """
        Carrega todos os dados
        """
        for lr_dir in sorted(self.base_dir.glob("lr_*")):
            # Itera sobre todos os diretórios que começam com "lr_" (ex: lr_0_1)
            # Usa sorted para garantir a ordem consistente na leitura
            lr_key = lr_dir.name.replace("lr_", "")
            # Extrai a chave do conjunto de pesos, removendo o prefixo "lr_"
            # Por exemplo, "lr_0_1" vira "0_1"
            for run_dir in sorted(lr_dir.glob("run_*")):
                # Itera sobre os diretórios de execução (run_*) dentro de cada diretório de pesos
                # Cada run representa uma execução ou experimento diferente
                run_index = int(run_dir.name.replace("run_", ""))
                # Extrai o índice da run (ex: "run_2" vira 2)
                record = {
                    "lr_init": lr_key, # Salva a chave de inicialização dos lr
                    "run": run_index, # Salva o núemro de execuções
                }

                best_model_file = next(run_dir.glob("best_model_epoch_*.json"), None)
                # Procura o arquivo JSOn com o melhor modelo salvo (com nome tipo: best_model_epoch_x.json)
                # Se não encontrar, est_model_file fica como None
                if best_model_file:
                    # Se encontrar arquivo com o melhor modelo 
                    match = re.search(
                        r"best_model_epoch_(\d+)\.json", best_model_file.name
                    ) # Extrai o número da época (epoch) do nome do arquivo usando regex
                    if match:
                        record["best_model_epoch"] = int(match.group(1))
                        # Salva a época em que o melhor modelo foi salvo
                    with open(best_model_file, "r") as f:
                        record["best_model"] = json.load(f)
                        # Abre o arquivo JSON e carrega os dados do melhor modelo para dicionário 'record'

                model_file = run_dir / "model.json"
                # Define o caminho para o arquivo com o modelo final da run
                if model_file.exists():
                    # Verifica se o arquivo do modelo final existe
                    with open(model_file, "r") as f:
                        record["final_model"] = json.load(f)
                        # Carrega o modelo final da run e armazena no dicionário 'record'

                test_report_file = run_dir / "test_report.json"
                # Define o caminho para o relatório de teste da run
                if test_report_file.exists():
                    # Verifica se o arquivo de teste existe
                    with open(test_report_file, "r") as f:
                        test_data = json.load(f)
                        # carrega os dados do relatório de teste
                        record.update(
                            {
                                "accuracy": test_data.get("accuracy"),
                                "avg_cross_entropy_loss": test_data.get(
                                    "avg_cross_entropy_loss"
                                ),
                                "total_samples": test_data.get("total_samples"),
                                "predictions": test_data.get("predictions"),
                            }
                        )

                validation_file = run_dir / "validation_log.json"
                # Define o caminho para o log de validação da run
                if validation_file.exists():
                    # Verifica se o arquivo de validação existe
                    with open(validation_file, "r") as f:
                        record["validation_log"] = json.load(f)
                        # Carrega o log de validação e armazena no dicionário 'Record'

                self.data.append(record)
                # Adiciona o dicionário 'record' completo (com dados da run)

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def save_dataframe(self, path=None):
        df = self.to_dataframe()
        path = path or self.base_dir / "compiled_data_test_2.pkl"
        df.to_pickle(path)

    def load_dataframe(self, path=None):
        path = path or self.base_dir / "compiled_data_test_2.pkl"
        return pd.read_pickle(path)


if __name__ == "__main__":
    analyzer_t1 = Analyzer_test_1()
    analyzer_t1.load_all()
    analyzer_t1.save_dataframe(path='./Reports/reports_test_1/compiled_data_test_1.pkl')
    analyzer_t2 = Analyzer_test_2()
    analyzer_t2.load_all()
    analyzer_t2.save_dataframe(path='./Reports/reports_test_2/compiled_data_test_2.pkl')
