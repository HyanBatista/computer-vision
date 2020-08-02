import numpy as np

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