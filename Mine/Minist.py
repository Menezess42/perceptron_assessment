import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical


# Carrega e prepara os dados
def load_and_prepare_data(path="./Reports/heart.csv"):
    df = pd.read_csv(path)
    X = df.drop("target", axis=1)
    y = df["target"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


# Treina e avalia uma arquitetura
def train_and_evaluate_model(name, model_fn, X_train, y_train, X_test, y_test):
    model = model_fn()
    history = model.fit(
        X_train, y_train, validation_split=0.2, epochs=50, batch_size=16, verbose=0
    )

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"[{name}] Accuracy: {acc:.4f} | Loss: {loss:.4f}")

    plot_training_curves(name, history)
    plot_confusion_matrix(name, model, X_test, y_test)


# Plota acurácia e perda
def plot_training_curves(name, history):
    plt.figure(figsize=(12, 5))

    # Acurácia
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Treino")
    plt.plot(history.history["val_accuracy"], label="Validação")
    plt.title(f"{name} - Acurácia")
    plt.xlabel("Épocas")
    plt.ylabel("Acurácia")
    plt.legend()

    # Perda
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Treino")
    plt.plot(history.history["val_loss"], label="Validação")
    plt.title(f"{name} - Perda")
    plt.xlabel("Épocas")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"./Reports/{name}_training.png")
    plt.close()


# Matriz de confusão
def plot_confusion_matrix(name, model, X_test, y_test):
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} - Matriz de Confusão")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.savefig(f"./Reports/{name}_confusion_matrix.png")
    plt.close()


# Arquiteturas diferentes
def simple_dense():
    model = Sequential(
        [
            Dense(16, activation="relu", input_shape=(13,)),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def deep_dense():
    model = Sequential(
        [
            Dense(32, activation="relu", input_shape=(13,)),
            Dropout(0.3),
            Dense(16, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def shallow_dropout():
    model = Sequential(
        [
            Dense(16, activation="relu", input_shape=(13,)),
            Dropout(0.2),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


# Executa os testes
def main():
    X_train, X_test, y_train, y_test = load_and_prepare_data()

    train_and_evaluate_model(
        "SimpleDense", simple_dense, X_train, y_train, X_test, y_test
    )
    train_and_evaluate_model("DeepDense", deep_dense, X_train, y_train, X_test, y_test)
    train_and_evaluate_model(
        "ShallowDropout", shallow_dropout, X_train, y_train, X_test, y_test
    )


if __name__ == "__main__":
    main()
