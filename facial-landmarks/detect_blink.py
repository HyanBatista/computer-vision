from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def eye_aspect_ratio(eye):
    # Calcule a distância euclidiana entre os dois conjuntos de coordenadas
    # dos marcos verticais do olho.
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # Calcule a distância euclidiana entre as coordenadas dos marcos horizontais
    # do olho.
    C = dist.euclidean(eye[0], eye[3])

    # Calcule o EAR.
    ear = (A + B) / (2.0 * C)

    # Retorne o eye aspect ratio.
    return ear

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())

"""
Defina duas constantes. Uma delas será para o eye aspect ratio que indica uma piscada.
Se o EAR ficar abaixo desse threshold e depois acima dele, um blink será registrado.
A outra constante será para o número de frames consecutivos em que o EAR deve estar 
abaixo do threshold para que seja registrado um blink.
"""
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 5

"""
Inicialize o contador de frames e o contador do número total de blinks. O COUNTER representa
o número de frames consecutivos em que o EAR ficou abaixo do threshold. Já o TOTAL
representa a quantidade de blinks detectados durante a executação do script.
"""
COUNTER = 0
TOTAL = 0

# Inicialize o detector de faces do dlib (baseado em HOG) e o facial landmark predictor.
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Pegue os índices dos marcos faciais dos olhos.
l_start, l_end = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
r_start, r_end = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Inicie a thread de stream de vídeo.
# vs = FileVideoStream(args["video"]).start()
# file_stream = True
vs = VideoStream().start()
# file_stream = True
time.sleep(1.0)

# Percorra os frames da stream de vídeo.
while True:
    """
    Se esta é uma stream de vídeo de arquivo, então nós precisamos checar se
    há mais frames sobrando no buffer para processar.
    """
    """
    if file_stream and not vs.more():
        break
    """
    """
    Capture o frame da threaded video file stream, redimensione-o e o converta
    para grayscale.
    """
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte faces no frame em grayscale.
    rects = detector(gray, 0)

    # Percorra as detecções.
    for rect in rects:
        """
        Determine os marcos faciais da região da face, depois converta as coordenadas (x, y)
        desses marcos para um NumPy array.
        """
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        # Extraia a coordenada do olho direito e esquerdo, depois as use para calcular os EAR de ambos.
        left_eye = shape[l_start:l_end]
        right_eye = shape[r_start:r_end]
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)

        """
        Faça a média da EAR dos dois olhos. Isso proporcionará uma melhor estimativa de blink, isso
        assumindo que um pessoa pisca os dois olhos ao mesmo tempo.
        """
        ear = (left_ear + right_ear) / 2.0

        # Calcule o casco convexo dos dois olhos e depois os visualize.
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)

        # Cheque se o eye aspect ratio está abaixo do blink threshold, se sim, incremente o blink frame counter.
        if ear < EYE_AR_THRESH:
            COUNTER += 1

        # Caso contrário, o eye aspect ratio não está abaixo do blink threshold.
        else:
            # Se os olhos ficaram fechados por um número suficiente de frames, incremente o número total de blinks.
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                TOTAL += 1

            # Resete o contador de frames do olho.
            COUNTER = 0
        
        """
        Desenhe o número total de blinks no frame junto com o eye aspect ratio
        calculado para o frame.
        """
        cv2.putText(
            img=frame,
            text="BLINKS: {}".format(TOTAL),
            org=(10, 30),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.7,
            color=(0, 0, 255),
            thickness=2
        )

        cv2.putText(
            frame,
            "EAR: {:.2f}".format(ear),
            (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

    # Mostre o frame.
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    # Se a tecla "q" for apertada, quebre o loop
    if key == ord("q"):
        break


# Limpe a memória.
cv2.destroyAllWindows()
vs.stop()
