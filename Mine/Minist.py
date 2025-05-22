# exercise3_mnist.py
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models, utils, datasets

def garante_pasta(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_mnist():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test  = x_test.astype('float32')  / 255.
    x_train = np.expand_dims(x_train, -1)
    x_test  = np.expand_dims(x_test, -1)
    y_train = utils.to_categorical(y_train, 10)
    y_test  = utils.to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

def build_mlp(input_shape=(28,28,1), num_classes=10):
    return models.Sequential([
        layers.Flatten(input_shape=input_shape),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])

def build_cnn(input_shape=(28,28,1), num_classes=10):
    return models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])

def train_model(model, x_train, y_train, name, output_dir,
                epochs=20, batch_size=128, lr=0.001):
    garante_pasta(output_dir)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=2
    )
    # salvar histórico para plot
    np.save(os.path.join(output_dir, f"{name}_history.npy"), history.history)
    # salvar pesos finais (opcional)
    model.save(os.path.join(output_dir, f"{name}_model.h5"))
    return history.history

def plot_history(hist, name, output_dir):
    garante_pasta(output_dir)
    epochs = range(1, len(hist['loss']) + 1)

    # Loss
    plt.figure(figsize=(6,4))
    plt.plot(epochs, hist['loss'],   'b-', label='Train Loss')
    plt.plot(epochs, hist['val_loss'], 'r--', label='Val Loss')
    plt.title(f'{name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_loss.png"))
    plt.close()

    # Accuracy
    plt.figure(figsize=(6,4))
    plt.plot(epochs, hist['accuracy'],   'b-', label='Train Acc')
    plt.plot(epochs, hist['val_accuracy'], 'r--', label='Val Acc')
    plt.title(f'{name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{name}_accuracy.png"))
    plt.close()

def evaluate_and_plot(models_info, x_test, y_test, output_dir):
    names, histories = zip(*models_info.items())
    final_acc = []
    final_loss = []

    for name, hist in histories:
        loss, acc = tf.keras.models.load_model(
            os.path.join(output_dir, name, f"{name}_model.h5")
        ).evaluate(x_test, y_test, verbose=0)
        final_loss.append(loss)
        final_acc.append(acc)

    # Gráfico comparativo
    x = np.arange(len(names))
    width = 0.35

    plt.figure(figsize=(6,4))
    plt.bar(x - width/2, final_acc, width, label='Accuracy')
    plt.bar(x + width/2, final_loss, width, label='Loss')
    plt.xticks(x, names)
    plt.ylabel('Valor')
    plt.title('Comparação Final: Accuracy vs Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison.png"))
    plt.close()

def main():
    base = './Reports/exercise3'
    garante_pasta(base)

    x_train, y_train, x_test, y_test = load_mnist()

    models_info = {}
    # MLP
    mlp = build_mlp()
    hist_mlp = train_model(mlp, x_train, y_train, 'mlp', os.path.join(base, 'mlp'))
    plot_history(hist_mlp, 'mlp', os.path.join(base, 'mlp'))
    models_info['mlp'] = ( 'mlp', hist_mlp )

    # CNN
    cnn = build_cnn()
    hist_cnn = train_model(cnn, x_train, y_train, 'cnn', os.path.join(base, 'cnn'))
    plot_history(hist_cnn, 'cnn', os.path.join(base, 'cnn'))
    models_info['cnn'] = ( 'cnn', hist_cnn )

    # Comparativo final
    evaluate_and_plot(models_info, x_test, y_test, base)

if __name__ == "__main__":
    main()
