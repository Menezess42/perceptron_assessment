import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import random
from sklearn.metrics import confusion_matrix


class Analyzer_test_1:
    def __init__(self, df):
        self.df = df

    def plot_stability_vs_weight_init(self, output_path="./Reports/reports_test_1/"):
        """
        Gera gráficos de estabilidade para accuracy e cross-entropy loss
        em função dos pesos iniciais (weight_init)

        Args:
            output_path (str): Caminho para pasta de salvamento do gráfico
        """
        df = self.df.copy()
        df["weight_float"] = df["weight_init"].apply(self.parse_weight)

        grouped = (
            df.groupby("weight_float")
            .agg(
                {"accuracy": ["mean", "std"], "avg_cross_entropy_loss": ["mean", "std"]}
            )
            .reset_index()
        )

        non_random = grouped[grouped["weight_float"] != "random"]
        random_row = grouped[grouped["weight_float"] == "random"]
        non_random = non_random.sort_values("weight_float")
        grouped_sorted = pd.concat([non_random, random_row])

        x_labels = grouped_sorted["weight_float"].astype(str)
        acc_mean = grouped_sorted[("accuracy", "mean")]
        acc_std = grouped_sorted[("accuracy", "std")]
        loss_mean = grouped_sorted[("avg_cross_entropy_loss", "mean")]
        loss_std = grouped_sorted[("avg_cross_entropy_loss", "std")]
        y_ticks_acc = np.round(np.unique(np.concatenate([acc_mean, np.linspace(acc_mean.min(), acc_mean.max(), 5)])), 5)
        y_ticks_loss = np.round(np.unique(np.concatenate([loss_mean, np.linspace(loss_mean.min(), loss_mean.max(), 5)])), 5)


        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.errorbar(x_labels, acc_mean, yerr=acc_std, fmt="o-", capsize=5)
        plt.title("Accuracy por Peso")
        plt.xlabel("Pesos iniciais")
        plt.ylabel("Accuracy média")
        plt.xticks(rotation=45)
        plt.grid(True)

        # Loss
        plt.subplot(1, 2, 2)
        plt.errorbar(x_labels, loss_mean, yerr=loss_std, fmt='o-', capsize=5, color="orange")
        plt.title("Cross-Entropy Loss por weight_init")
        plt.xlabel("Pesos iniciais (weight_init)")
        plt.ylabel("Loss média")
        plt.xticks(rotation=45)
        plt.yticks(y_ticks_loss) 
        plt.grid(True)

        plt.tight_layout()

        output_path += "plot_de_estabilidade.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()

        print(f"Gráfico salvo em: {output_path}")

    def parse_weight(self, val):
        if val == "random":
            return "random"
        try:
            return float(val.replace("_", "."))
        except:
            return val

    def plot_confusion_subplots_by_weight(self, output_path="./Reports/reports_test_1/"):
        """
        Gera subplots com 4 matrizes de confusão aleatórias para cada valor de weight_init.

        Args:
            output_path (str): Pasta para salvar os plots.
        """
        df = self.df.copy()
        pesos_unicos = df["weight_init"].unique()

        df["weight_float"] = df["weight_init"].apply(self.parse_weight)

        pesos_unicos = df["weight_float"].unique()

        for peso_float in sorted(pesos_unicos, key=lambda x: float(x) if x != "random" else float("inf")):
                df_peso = df[df["weight_float"] == peso_float]

                runs_disponiveis = df_peso["run"].unique()
                runs_selecionadas = random.sample(list(runs_disponiveis), min(4, len(runs_disponiveis)))

                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                fig.suptitle(f"Matrizes de Confusão - Peso: {peso_float}", fontsize=16)

                for i, run in enumerate(runs_selecionadas):
                    ax = axes[i // 2, i % 2]
                    dados_run = df_peso[df_peso["run"] == run]["predictions"].values[0]

                    y_true = [d["true_class"] for d in dados_run]
                    y_pred = [d["predicted_class"] for d in dados_run]

                    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                                xticklabels=[0, 1, 2], yticklabels=[0, 1, 2], ax=ax)

                    ax.set_title(f"Run {run}")
                    ax.set_xlabel("Predito")
                    ax.set_ylabel("Real")

                plt.tight_layout(rect=[0, 0, 1, 0.95])
                filename = f"{output_path}/confusion_{peso_float}.png"
                plt.savefig(filename)
                plt.close()

class Analyzer_test_2:
    def __init__(self, df):
        self.df = df

    def parse_lr(self, val):
        try:
            return float(val.replace("_", "."))
        except:
            return val

    def plot_stability_vs_lr(self, output_path="./Reports/reports_test_2/"):
        """
        Gera gráficos de estabilidade para accuracy e cross-entropy loss
        em função da taxa de aprendizado (learning rate).
        """

        df = self.df.copy()
        df["lr_float"] = df["lr_init"].apply(self.parse_lr)

        grouped = (
                df.groupby("lr_float")
                .agg({"accuracy": ["mean", "std"], "avg_cross_entropy_loss": ["mean", "std"]})
                .reset_index()
                )
        
        grouped_sorted = grouped.sort_values("lr_float")

        x_labels = grouped_sorted["lr_float"].astype(str)
        acc_mean = grouped_sorted[("accuracy", "mean")]
        acc_std = grouped_sorted[("accuracy", "std")]
        loss_mean = grouped_sorted[("avg_cross_entropy_loss", "mean")]
        loss_std = grouped_sorted[("avg_cross_entropy_loss", "std")]

        y_ticks_loss = np.round(np.unique(np.concatenate([
            loss_mean,
            np.linspace(loss_mean.min(), loss_mean.max(), 5)
            ])), 5)

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.errorbar(x_labels, acc_mean, yerr=acc_std, fmt="o-", capsize=5)
        plt.title("Accuracy por Learning Rate")
        plt.xlabel("Learning Rate")
        plt.ylabel("Accuracy Média")
        plt.xticks(rotation=45)
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.errorbar(x_labels, loss_mean, yerr=loss_std, fmt="o-", capsize=5, color="orange")
        plt.title("Cross-Entropy Loss por Learning Rate")
        plt.xlabel("Learning Rate")
        plt.ylabel("Loss média")
        plt.xticks(rotation=45)
        plt.grid(True)

        plt.tight_layout()

        output_file = output_path+"plot_estabilidade.png"
        plt.savefig(output_file, bbox_inches="tight")
        plt.close()

        print(f"Gráfico salvo em: {output_file}")

    def plot_confusion_subplots_by_lr(self, output_path="./Reports/reports_test_2/"):
        """
        Gera subplots com 4 matrizes de confusão aleatórias para cada valor de learning rate.
        """
        df = self.df.copy()
        df["lr_float"] = df["lr_init"].apply(self.parse_lr)
        lrs_unicos = df["lr_float"].unique()

        for lr_float in sorted(lrs_unicos):
            df_lr = df[df["lr_float"] == lr_float]

            runs_disponiveis = df_lr["run"].unique()
            runs_selecionadas = random.sample(list(runs_disponiveis), min(4, len(runs_disponiveis)))

            fig, axes = plt.subplots(2, 2, figsize=(10,8))
            fig.suptitle(f"Matrizes de confusão - Learning Rate: {lr_float}", fontsize=16)

            for i, run in enumerate(runs_selecionadas):
                ax = axes[i//2, i % 2]
                dados_run = df_lr[df_lr["run"] == run]["predictions"].values[0]
                y_true = [d["true_class"] for d in dados_run]
                y_pred = [d["predicted_class"] for d in dados_run]

                cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                            xticklabels=[0, 1, 2], yticklabels=[0, 1, 2], ax=ax)

                ax.set_title(f"Run {run}")
                ax.set_xlabel("Predito")
                ax.set_ylabel("Real")

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            filename = f"{output_path}/confusion_{lr_float}.png"
            plt.savefig(filename)
            plt.close()

            print(f"Matriz de confusão salva: {filename}")

def main():
    df1 = pd.read_pickle("./Reports/reports_test_1/compiled_data.pkl")
    a1 = Analyzer_test_1(df=df1)
    a1.plot_stability_vs_weight_init()
    a1.plot_confusion_subplots_by_weight()
    df2 = pd.read_pickle("./Reports/reports_test_2/compiled_data_test_2.pkl")
    a2 = Analyzer_test_2(df=df2)
    a2.plot_stability_vs_lr()
    a2.plot_confusion_subplots_by_lr()

if __name__ == "__main__":
    main()
