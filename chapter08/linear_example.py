import numpy as np
import cv2

# Inicialize os rótulos de classe e configure a seed do gerador de números
# pseudo-aleatórios para que possamos reproduzir nossos resultados.
labels = ["dog", "cat", "panda"]
np.random.seed(1)

# Inicialize aleatoriamente nossa matrix de pesos e vetor de vieses. -- em um
# tarefa de classificação e treino real, esses parametros seriam aprendidos
# pelo nosso modelo, mas para este exemplo, vamos usar valores aleatórios.
W = np.random.randn(3, 3072)
b = np.random.randn(3)

# Carregue nossa imagem de exemplo, redimensione-a, e achate-a em nossa representação
# de vetor de características.
orig = cv2.imread("beagle.png")
image = cv2.resize(orig, (32, 32)).flatten()

# Calcule os scores de saída fazendo o dot product entre a
# matrix de pesos e os pixéis da imagem e adicionando ao resultado
# as bias.
scores = W.dot(image) + b

# Itere sobre os scores + labels e mostre-os.
for (label, score) in zip(labels, scores):
    print("[INFO] {}: {:.2f}".format(label, score))

# Desenhe a label com a mais alta pontuação na imagem como nossa previsão.
cv2.putText(
    img=orig,
    text="Label: {}".format(labels[np.argmax(scores)]),
    org=(10, 30),
    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
    fontScale=0.9,
    color=(0, 255, 0),
    thickness=2
)

# Mostre a imagem de entrada.
cv2.imshow("Image", orig)
cv2.waitKey(0)