from pickletools import optimize
from matplotlib import widgets
import numpy as np
import progressbar
from src.utils.Plot import bar_widgets

class Network():
    """
    Modelo base dos algoritmos de deep learning.
    
    Parametros:
    - optimizer: class
    * O otimizador de peso que será usado para ajustar os pesos para minimizar a perda.
    - loss: class
    * Função de perda usada para medir o desempenho do modelo. (SquareLoss ou CrossEntropy).
    - validation: tuple
    * Tupla contendo dados de validação e seus rotulos (X, y)
    """
    def __init__(self, optimizer, loss, validation_data = None):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()
        self.progressbar = progressbar.ProgressBar(widgets = bar_widgets)
        
        self.val_set = None
        if validation_data:
            X, y = validation_data
            self.val_set = {"X": X, "y": y}
    
    def setTrainable(self, trainable):
        """
        Método que permite congelamento dos pesos das camadas da rede
        """
        for layer in self.layers:
            layer.trainable = trainable
    
    def add(self, layer):
        """
        Método que permite adicionar uma camada a rede.
        Se essa não for a primeira camada adicionada, então setar o input shape
        para o output shapa da ultima camada adicionada.
        """
        if self.layers:
            layer.setInputShape(shape = self.layers[-1].outputShape())
        
        # if a camada tiver pesos que precisam ser inicializados
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer = self.optimizer)
        
        # adicionar a camada na rede
        self.layers.append(layer)