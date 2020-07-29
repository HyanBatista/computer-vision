"""Importe os pacotes necessários"""
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        """ 
        Armazene o tamanho e a altura da imagem e, também, o método de interpolação usado
        para redimensionamento da imagem.
        """
        self.width = width
        self.height = height
        self.inter = inter

    def preprocess(self, image):
        """
        Redimensione a imagem para um tamanho fixo ignorando a proporção da tela.
        """
        return cv2.resize(image, (self.width, self.height), interpolation=self.inter)
        