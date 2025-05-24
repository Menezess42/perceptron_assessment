# Exercício 2: Wine

## orgainização:
### Wine.py:
- Código principal, responsável por criar a rede perceptron, criar a função de teste, validação, treino, shuffle, hot-Encoder, softMax, cross Entropy, e as funções experiment_1_testing_weightRange e experiment_2_testing_learning_rate que são as funções responsáveis por realizar os experimentos do peso e da taxa de aprendizado.
- Este experimento difere do da Iris por ser normalizado em limites de -1 a 1.

### analyzer_make_dataFrame.py:
- Código auxiliar, responsável por percorrer as pastas test_1 e test_2, consumir as informações, organizar e salvar em um dataframe pandas.

### analyzer_see_pickle_test_1.py
- Código auxiliar, responsável por consumir o dataFrame sobre test_1 e ajudar a visualizar os dados para poder desenhar os experimentos e compor o relatório.

### analyzer_see_pickle_test_2.py
- Código auxiliar, responsável por consumir o dataFrame sobre test_2 e ajudar a visualizar os dados para poder desenhar os experimentos e compor o relatório.

### analyzer_create_plot.py
- Código auxiliar, responsável por consumir os dataFrames e plotar as devidas métricas para o relatório.

### Pasta DataSet:
- Contém os dados do dataSet Iris
### Pasta Reports
- Esta pasta é divida em 4 subpastas:
    - test_1
    - test_2
    - reports_test_1
    - reports_test_2
#### test_1:
- Contém 11 pastas. 10 pastas vão de weights_0_1 a weights_1_0 que representam os testes com pesos de 0.1 a 1.0. E a pasta weights_random que representa a pasta onde os pesos inicias foram aleatórios.
    - Estas pastas contém o mesmo núcleo. Pastas que vão de run_0 a run_9, e dentro dela temos os seguintes arquivos json:
        - best_model_epoch_X.json
            - Arquivo contendo o formato e os pesos do modelo que a função de validação julgou ser o melhor, onde X é a época.
        - model.json
            - Arquivo contendo o formato e os pesos do modelo após a execução das 100 épocas.
        - Test_report.json:
            - Arquivo contendo as métricas de acurácia e as predições do modelo(model.json).
        - validation_log.json
            - Arquivo conténdo as métricas de avalição da função de validação por época.
#### test_2:
- Contém 10 pastas vão de lr_0_01 a lr_0_1 e depois de lr_0_1 a lr_1_0, lr=Learning Rate, que representam os testes com taxas de aprendizado de 0.01 a 0.1 e depois de 0.1 a 1.0 ..
    - Estas pastas contém o mesmo núcleo. Pastas que vão de run_0 a run_9, e dentro dela temos os seguintes arquivos json:
        - best_model_epoch_X.json
            - Arquivo contendo o formato e os pesos do modelo que a função de validação julgou ser o melhor, onde X é a época.
        - model.json
            - Arquivo contendo o formato e os pesos do modelo após a execução das 100 épocas.
        - Test_report.json:
            - Arquivo contendo as métricas de acurácia e as predições do modelo(model.json).
        - validation_log.json
            - Arquivo conténdo as métricas de avalição da função de validação por época.
#### reports_test_1:
- Pasta conténdo:
    - compiled_data.pkl, um dataFrame pandas que reune os dados da pasta test_1. 
    - Confusion_X.png: Imagens png que mostram 4 matrizes de confusões para 4 runs de cada peso do teste 1.
    - plot_de_estabilidade.png: Imagem png que demosntra o comportamento do modelo em relação a acurácia média por peso e a perda por entropia cruzada média por peso.
#### reports_test_2:
- Pasta conténdo:
    - compiled_data_test_2.pkl, um dataFrame pandas que reune os dados da pasta test_2. 
    - Confusion_X.png: Imagens png que mostram 4 matrizes de confusões para 4 runs de cada taxa de aprendizado do teste 2.
    - plot_de_estabilidade.png: Imagem png que demosntra o comportamento do modelo em relação a acurácia média por taxa de aprendizado e a perda por entropia cruzada média por taxa de aprendizado.