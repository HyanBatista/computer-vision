import face_utils
import numpy as np
import argparse
import imutils
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())


# Inicialize o detector de faces do dlib e depois crie o facial landmark predictor.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Carregue a imagem de input, redimensione-a e a converta-a para grayscale.
image = cv2.imread(args["image"])
image = imutils.resize(image, width=500)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detecte faces na imagem em grayscale.
rects = detector(gray, 1)

# Percorra as detecções.
for (i, rect) in enumerate(rects):
    # Determine os marcos faciais para a região de interesse, depois
    # converta as coordenadas dos marcos para um NumPy array.
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    for (name, (i, j)) in face_utils.FACIAL_LANDMARKS_IDXS.items():
        # Clone a imagem original para que possamos desenhar nela.
        # Mostre o nome da parte do rosto na imagem.
        clone = image.copy()
        cv2.putText(
            img=clone,
            text=name,
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 225),
            thickness=2
        )

        # Percorra o subconjunto de facial landmarks a parte específica do rosto.
        for (x, y) in shape[i:j]:
            cv2.circle(
                img=clone,
                center=(x, y),
                radius=1,
                color=(0, 0, 255),
                thickness=-1
            )
        
        # Extraia a ROI da região da face como uma imagem separada.
        (x, y, w, h) = cv2.boundingRect(np.array([shape[i:j]]))
        roi = image[y:y + h, x:x + w]
        roi = imutils.resize(roi, width=250, inter=cv2.INTER_CUBIC)

        # Mostre a parte específica da face.
        cv2.imshow("ROI", roi)
        cv2.imshow("Image", clone)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # Visualize todas os marcos faciais com um revestimento transparente.

    output = face_utils.visualize_facial_landmarks(image, shape)
    cv2.imshow("Image", output)
    cv2.waitKey(0)  