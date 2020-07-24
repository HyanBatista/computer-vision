import numpy as np 
import argparse
import cv2


# construa o analisador de argumentos e analise os argumentos 
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
     help="path to input image")
ap.add_argument("-p", "--prototxt", required=True, 
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True, 
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5, 
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# carregue o serialized model a partir do disco
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# carregue uma imagem de entrada e construa um input blob para a imagem
# redimensionando para um tamanho fixo de 300x300 píxeis e depois normalizando
image = cv2.imread(args["image"])
(h, w) = image.shape[:2]
# efetue mean subtraction e scaling
blob = cv2.dnn.blobFromImage(
    cv2.resize(image, (300, 300)), 
    scalefactor=1.0, 
    size=(300, 300), mean=(104.0, 177.0, 123.0)
    )

#passe a imagem através da rede e obtenha as detecções e previsões
print("[INFO] computing object detections...")
net.setInput(blob)
detections = net.forward()

print("[INFO] detections shape {}".format(detections.shape[2]))

count = 0
# itere sobre as detecções
for i in range(0, detections.shape[2]):
    # extraia a certeza (i.e., probabilidade) associada com a previsão
    confidence = detections[0, 0, i, 2]
    
    # filtre as detecções fracas assegurando que a certeza é maior do que a certeza mínima
    if confidence > [args['confidence']]:
        # compute as coordenadas (x, y) da caixa delimitadora para o objeto
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # desenhe a caixa delimitadora da face junto com a probabilidade associada
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 225), 2)
        cv2.putText(
            image, 
            text, 
            (startX, y), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.45, 
            (0, 0, 255), 
            2
        )
    
    count += 1

cv2.imshow("Output", image)
print("[INFO] number of loops: ", count)
cv2.waitKey(0)
