import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
from sklearn.metrics import confusion_matrix

class ExperimentAnalyzer:
    def __init__(self, base_path: str):
        self.base_path = base_path
        self.data = []
        self.accuracy_data = []
        self.validation_data = {}
        self.confusion_data = {}
        self._load_data()

    def _load_data(self):
        for config in os.listdir(self.base_path):
            config_path = os.path.join(self.base_path, config)
            if not os.path.isdir(config_path):
                continue
            for run in os.listdir(config_path):
                run_path = os.path.join(config_path, run)
                if not os.path.isdir(run_path):
                    continue
                try: 
                    with open(os.path.join(run_path, 'test_report.json')) as f:
                        test_report = json.load(f)
                    with open(os.path.join(run_path, "validation_log.json")) as f:
                        val_log = json.load(f)

                except FileNotFoundError:
                    continue

                acc = test_report.get("accuracy")
                mse = test_report.get("avg_mse_loss")
                cel = test_report.get("avg_cross_entropy_loss")
                best_model = self._get_best_model_epoch(run_path)

                self.data.append({
                    "config": config,
                    "run": run,
                    "accuracy": acc,
                    "mse": mse,
                    "cross_entropy": cel,
                    "best_epoch": best_model.get("epoch", None)
                    })

                # Armazena curva de validação
                self.validation_data.setdefault(config, []).append(val_log)

                # Armarzena informação para confusion matrix
                y_true = [p["true_class"] for p in test_report["predicitions"]]
                y_pred = [p["predicted_class"] for p in test_report["predicitions"]]

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
                self.confusion_data.setdefault(config, []).append(cm)

    def _get_best_model_epoch(self, run_path: str):
        for file in os.listdir(run_path):
            if file.startswith("best_model_epoch_") and file.endswith(".json"):
                epoch = int(file.replace("best_model_epoch_", "").replace(".json", ""))
                with open(os.path.join(run_path, file)) as f:
                    model_data = json.load(f)

                model_data["epoch"] = epoch
                return model_data
        return {"epoch": None}

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.data)

    def plot_accuracy_boxplot(self):
        df = self.to_dataframe()
        plt.figure(figsize=(10, 6))
        sns.boxplot(x="config", y="accuracy", data=df)
        plt.xticks(rotation=45)
        plt.title("Boxplot de acurácia por Configuração de Peso")
        plt.tight_layout()
        plt.show()
        plt.savefig("./Reports/reports_test_1/boxplot_de_acuracia_por_configuracao_de_peso.png")

    def plot_loss_curves(self):
        plt.figure(figsize=(12, 6))
        for config, runs in self.validation_data.items():
            for run_log in runs:
                epochs = [e["epoch"] for e in run_log]
                cel = [e["cross_entropy_loss"] for e in run_log]
                mse = [e["mse_loss"] for e in run_log]
                plt.plot(epochs, cel, label=f"{config} - CE", alpha=0.3)
                plt.plot(epochs, mse, label=f"{config} - MSE", alpha=0.3)

        plt.title("Curvas de Perda por Época")
        plt.xlabel("Época")
        plt.ylabel("Loss")
        plt.legend(loc='upper right', fontsize="small", ncol=2)
        plt.tight_layout()
        plt.show()
        plt.savefig("./Reports/reports_test_1/curvas_de_perda_por_epoca.png")

    def plot_accuracy_bar(self):
        df = self.to_dataframe()
        summary = df.groupby("config").agg(
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std")
                ).reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='config', y='mena_accuracy', yerr=summary['std_accuracy'], data=summary)
        plt.xticks(rotation=45)
        plt.title("Acurácia Méida e Desvio Padrão por Configuração")
        plt.tight_layout()
        plt.show()
        plt.savefig("./Reports/reports_test_1/Acuracia_media_e_desvio_adrao_por_configuracao.png")


    def plot_best_epoch_bar(self):
        df = self.to_dataframe()
        epoch_summary = df.groupby("config")["best_epoch"].mean().reset_index()
        plt.figure(figsize=(10, 6))
        sns.barplot(x='config', y='best_epoch', data=epoch_summary)
        plt.xticks(rotation=45)
        plt.title('Época Média do Melhor Modelo por Configuração')
        plt.tight_layout()
        plt.show()
        plt.savefig("./Reports/reports_test_1/Epoca_Media_do_Melhor_Modelo_por_Configuracao.png")
    
    def plot_confusion_heatmap(self):
        avg_confusions = {}
        for config, matrices in self.confusion_data.items():
            avg_confusions[config]=np.mean(matrices, axis=0)

        num_config = len(avg_confusions)
        fig, axes = plt.subplots(1, num_config, figsize=(5*num_config, 4))
        if num_config == 1:
            axes = [axes]
        for ax, (config, cm) in zip(axes, avg_confusions.items()):
            sns.heatmap(cm, annot=True, fmt=".0f", cmap="Blues", ax=ax)
            ax.set_title(f"Matriz de Confusão: {config}")
            ax.set_xlabel("Classe Predita")
            ax.set_ylabel("Classe Real")

        plt.tight_layout()
        plt.show()
        plt.savefig("./Reports/reports_test_1/heatmap_matriz_de_confusao.png")

    def plot_accuracy_vs_loss(self):
        df = self.to_dataframe()
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x="cross_entropy", y="accuracy", hue="config", data=df)
        plt.tight_layout()
        plt.show()
        plt.savefig("./Reports/reports_test_1/accuracy_vs_loss.png")

    
if __name__ == "__main__":
    analyzer = ExperimentAnalyzer("./Reports/test_1")
    analyzer.plot_accuracy_boxplot()



