import argparse
import numpy as np
import dlib
import cv2
import imutils
from imutils import face_utils

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

# Inicialize o detector de faces do dlib (HOG-based) e crie o preditor de facial landmarks.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Carregue a imagem de input, redimensione-a e converta-a para grayscale.
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecte faces na imagem em grayscale.
# O segundo parâmetro diz respeito ao número de pyramid layers da imagem para aplicar
# quando aumentando a priori da imagem para aplicar o detector.
# Aumentar o input image prior nos permite detectar mais faces, porém a custo computacional maior.
rects = detector(gray, 1)


# Intere sobre as detecções de face
for (i, rect) in enumerate(rects):
    # Determine os marcos faciais para a região da face, então converta as coordenadas-(x, y)
    # do marco facial para um array NumPy.
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    # Converta o retângulo da dlib para uma bouding box no estilo OpenCV
    # [i.e., (x, y, w, h)], depois desenhe a bouding box da face.
    (x, y, w, h) = face_utils.rect_to_bb(rect)
    cv2.rectangle(
        img=image,
        pt1=(x, y),
        pt2=(x + w, y + h),
        color=(0, 255, 0),
        thickness=2
    )

    # Mostre o número da face.
    cv2.putText(
        img=image,
        text="Face #{}".format(i + 1),
        org=(x - 10, y - 10),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=(0, 255, 0),
        thickness=2
    )

    # Itere sobre as coordenadas-(x, y) das facial landmarks e as desenhe na imagem.
    for (x, y) in shape:
        cv2.circle(
            img=image,
            center=(x, y),
            radius=1,
            color=(0, 0, 255),
            thickness=-1
        )

# Mostre imagem de output com as face detections e a facial landmarks.
cv2.imshow("Output", image)
cv2.waitKey(0)