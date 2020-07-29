"""Importe os pacotes necessários"""
import numpy as np
import cv2
import os

class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        """Armazene os pré-processadores de imagens"""
        self.preprocessors = preprocessors

        """Se os pré-processadores estão None, os inicie com uma lista vazia"""
        if self.preprocessors is None:
            self.preprocessors = []
    
    def load(self, image_paths, verbose=-1):
        """Inicie a lista de features e labels"""
        data, labels = [], []

        """Itere sobre as imagens de entrada"""
        for (i, image_path) in enumerate(image_paths):
            """
            Carregue a image e extraia a class label assumindo que nosso path tem o
            seguinte formato: /path/to/dataset/{class}/{image}.jpg
            """
            image = cv2.imread(image_path)
            label = image_path.split(os.path.sep)[-2]

            """Verifique se os pré-processadores não estão None"""
            if self.preprocessors is not None:
                """Itere sobre os pré-processadores e aplique cada um à imagem"""
                for p in self.preprocessors:
                    image = p.preprocess(image)
            
            """
            Trate a imagem processada como um "feature vector" atualizando 
            a lista de dados e em seguida a lista de labels
            """
            data.append(image)
            labels.append(label)

            """Mostre uma atualização de cada 'verbose' imagens"""
            if verbose > 0 and i > 0 and (i + 1) % verbose == 0:
                print(f"[INFO] processed {i + 1}/{len(image_paths)}")

        return (np.array(data), np.array(labels))
