import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from scipy import stats


class PlotCreator:
    def __init__(self, df: pd.DataFrame, output_dir: str = "./Reports/reports_test_1/"):
        self.df = df
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def boxplot_accuracy_by_weight_config(self):
        # Criar rótulos bonitos para o eixo X
        self.df["weight_label"] = self.df["weight_init"].apply(
            lambda x: "random" if x == "random" else str(float(x))
        )

        plt.figure(figsize=(12, 6))
        sns.boxplot(
            data=self.df,
            x="weight_label",
            y="accuracy",  # Corrigido aqui
            palette="Set3"
        )
        plt.title("Boxplot de Acurácia por Configuração de Pesos Iniciais")
        plt.xlabel("Configuração de Peso Inicial")
        plt.ylabel("Acurácia no Teste")
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, "boxplot_accuracy_by_weight_config.png")
        plt.savefig(output_path)
        plt.close()
        print(f"[✔] Gráfico salvo em: {output_path}")

    def plot_loss_curves_by_weight_config(self, loss_type="cross_entropy"):
        """
        loss_type: "cross_entropy" ou "mse"
        """
        loss_type = "cross_entropy_loss"
        plt.figure(figsize=(12, 6))

        # Extrair dados por configuração de peso
        grouped = self.df.groupby("weight_init")

        for weight_value, group in grouped:
            all_runs_losses = []

            for _, row in group.iterrows():
                log = row["validation_log"]
                losses = [entry[loss_type] for entry in log]
                all_runs_losses.append(losses)

            # Padronizar comprimento (caso alguma run tenha menos épocas)
            min_len = min(len(run) for run in all_runs_losses)
            all_runs_losses = [run[:min_len] for run in all_runs_losses]

            # Transpor para obter lista de perdas por época
            losses_by_epoch = np.array(all_runs_losses).T  # shape: (n_epochs, n_runs)

            mean_loss = losses_by_epoch.mean(axis=1)
            sem = stats.sem(losses_by_epoch, axis=1)
            conf_interval = 1.96 * sem

            epochs = np.arange(1, len(mean_loss) + 1)

            label = str(weight_value)
            if isinstance(weight_value, str) and not weight_value.replace('.', '', 1).isdigit():
                label = "random"

            plt.plot(epochs, mean_loss, label=f"{label}")
            plt.fill_between(epochs, mean_loss - conf_interval, mean_loss + conf_interval, alpha=0.2)

        plt.title("Curvas de Perda por Época por Configuração de Peso Inicial")
        plt.xlabel("Época")
        ylabel = "Perda Cross-Entropy" if loss_type == "cross_entropy" else "Erro Quadrático Médio"
        plt.ylabel(ylabel)
        plt.legend(title="Peso Inicial")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig("plot_loss_curves_by_weight_config.png")
        plt.show()
                


if __name__ == '__main__':
    df = pd.read_pickle('./Reports/reports_test_1/compiled_data.pkl')
    #print(df.head())
    print(df.columns.tolist())
    print(df['weight_init'])
    pc = PlotCreator(df=df)
    #pc.boxplot_accuracy_by_weight_config() 
    pc.plot_loss_curves_by_weight_config()
