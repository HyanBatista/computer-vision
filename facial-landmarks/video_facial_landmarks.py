from imutils.video import VideoStream
from imutils import face_utils
import datetime
import argparse
import imutils
import time
import dlib
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
	help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())

# Inicialize o detector de faces da biblioteca dlib (baseado em HOG) e então crie 
# o facial landmark predictor.
# HOG: histograma de gradientes orientados
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# Inicialize a VideoStream e deixe o sensor da câmera aquecer um pouco.
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)

# Percorra os frames da stream de vídeo.
while True:
    # Capiture o frame da threaded video stream, redimensione-o para ter um tamanho 
    # máximo de 400 píxeis e o converta para grayscale.
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecte faces no frame em grayscale.
    rects = detector(gray, 0)

    # Percorra as detecções de face
    for rect in rects:
        # Determine os marcos faciais para cada região da face, depois converta as
        # coordenadas das landmarks em um NumPy array.
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # Percorra as coordenadas dos marcos faciais e os desenhe na imagem.
        for x, y in shape:
            cv2.circle(img=frame, center=(x, y), radius=1, color=(0, 0, 255), thickness=-1)
    
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # Se o "q" for pressionado, "quebre" o loop.
    if key == ord("q"):
        break

# Limpe a memória.
cv2.destroyAllWindows()
vs.stop()
