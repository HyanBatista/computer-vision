import numpy as np
from collections import OrderedDict
import cv2

# rect representa uma bounding box (que contém as coordenadas-(x, y) da detecção) 
# produzida por um dlib detector.
def rect_to_bb(rect):
    # Pegue uma delimitação prevista pelo dlib e a converta para o formato (x, y, w, h)
    # compatível com o OpenCV
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y


    # Retorne uma tupla de (x, y, w, h)
    return (x, y, w, h)

# shape é um objeto que contém as 68 coordenadas das regiões de 
# marcos faciais que são retornadas pelo dlib facial landmark detector

def shape_to_np(shape, dtype="int"):
    # Inicialize a lista de coordenadas-(x, y).
    coords = np.zeros((68, 2), dtype=dtype)

    # Itere sobre as 68 facial landmarks e as converta para uma 2-tuple de coordenadas-(x, y)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    
    # Retorne a lista de coordenadas-(x, y)
    return coords

# Defina um dicionário que conecte os índices dos marcos faciais à regiões específicas da face
FACIAL_LANDMARKS_IDXS = OrderedDict(
    ("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 35)),
	("jaw", (0, 17))
)

def visualize_facial_landmarks(image, shape, colors=None, alpha=0.75):
    """
    image: a imagem na qual serão desenhadas as facial landmarks.
    shape: o array numpy com as 68 coordenadas (x, y) que estão conectadas com várias estruturas faciais.
    color: uma lista de tuplas BGR usadas para colorir (color-code) cada umas das facial landmark regions.
    alpha: um parâmetro para controlar a opacidade do revestimento (overlay) na imagem original.
    """
    # Crie duas copias da imagem de input -- uma para o revestimento e outra para a imagem de output final.
    # O objetivo aqui é desenhar um revestimento semi transparente na imagem de output.
    overlay = image.copy()
    output = image.copy()

    # Se a lista de cores está vazia (None), inicialize-a com uma cor única para cada região de marcos faciais.
    if color is None:
        colors = [
            (19, 199, 109), 
            (79, 76, 240), 
            (230, 159, 23), 
            (168, 100, 168), 
            (158, 163, 32), 
            (163, 38, 32),
            (180, 42, 220)
        ]
    
    # Percorra individualmente as facial landmark regions.
    for (i, name) in enumerate(FACIAL_LANDMARKS_IDXS.keys()):
        # Pega as coordenadas associadas com o marco de face (face landmark).
        (j, k) = FACIAL_LANDMARKS_IDXS[name]
        pts = shape[j:k]

        # Verifique se é para desenhar a linha do queixo.
        if name == "jaw":
            # Já que o queixo não é uma região facial fechada, apenas desenhe linhas
            # entre as coordenadas (x, y).
            for l in range(1, len(pts)):
                ptA = tuple(pts[l - 1])
                ptB = tuple(pts[l])
                cv2.line(
                    overlay,
                    ptA,
                    ptB,
                    colors[i],
                    2
                )
            
        # Caso contrário calcule casco convexo das coordenadas e o exiba.
        else:
            hull = cv2.convexHull(pts)
            cv2.drawContours(
                overlay,
                [hull],
                -1,
                colors[i],
                -1
            )
    # Aplique o revestimento transparente (transparent overlay).
    cv2.addWeighted(
        src1=overlay,
        alpha=alpha,    
        src2=output,
        beta=1 - alpha,
        gamma=0,
        dst=output
    )