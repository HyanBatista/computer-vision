import numpy as np
"""
Um simples perceptron escrito em Python. Ele usa como função de ativação uma step funcion
e pode ser treinado usando um bitwise dataset.
"""

class Perceptron:
    def __init__(self, N, alpha=0.1):
        # Inicialize a matrix de pesos e armazene a taxa de aprendizado.
        # Dividir a weight matrix pela raiz quadrada do número de inputs
        # é uma técnica comum para escalar a weight matrix de forma a se
        # ter uma convergência mais rápida.
        self.W = np.random.randn(N + 1) / np.sqrt(N)
        self.alpha = alpha

    def step(self, x):
        # Aplique a step function.
        return 1 if x > 0 else 0
    
    def fit(self, X, y, epochs=10):
        # Insira uma coluna de 1's como a última entrada na matriz de 
        # características -- este truque nos permite tratar o viés
        # como um parâmetro treinável dentro de uma matriz de pesos.
        X = np.c_[X, np.ones( (X.shape[0]) )]

        # Itere sobre o número desejado de épocas.
        for epochs in np.arange(0, epochs):
            #Itere sobre cada data point.
            for (x, target) in zip(X, y):
                # Faça o produto escalar entre as input features
                # e a weight matrix, depois passe o resultado 
                # para a step function para obter a previsão.
                p = self.step( np.dot(x, self.W) )

                # Só efetue uma atualização nos pesos se a previsão não for igual
                # à output target class label.
                if p != target:
                    # Determine o erro.
                    error = p - target

                    # Atualize a weight matrix.
                    self.W += -self.alpha * error * x

    def predict(self, X, add_bias=True):
        # Se assegure de que o input é uma matriz.
        X = np.atleast_2d(X)

        # Verifique se é para adicionar a coluna de vieses.
        if add_bias:
            # Insira a coluna de 1's como sendo a última coluna da matriz (bias)
            X = np.c_[X, np.ones( (X.shape[0]) )]
            

        # Faça o produto escalar entre a o vetor de features de entrada e a weight matrix, depois
        # passe o valor para a step function.
        return self.step( np.dot(X, self.W) )
