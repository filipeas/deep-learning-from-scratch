from Layer import Layer
import math
import numpy as np
import copy

class Dense(Layer):
    """
    Camada totalmente conectada.
    
    Parametros:
    - n_units: int
        * Número de neuronios da camada.
    - input_shape: tuple
        * input shape esperado para a camada. Para camada densa um
        unico digito especifica o numero de caracteristicas da entrada.
        Deve ser especificado se é a primeira camada na rede.
    """

    def __init__(self, n_units, input_shape = None):
        self.layer_input = None
        self.input_shape = input_shape
        self.n_units = n_units
        self.trainable = True
        self.W = None
        self.w0 = None
    
    def initialize(self, optimizer):
        # inicializa os pesos
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.w0 = np.zeros((1, self.n_units))
        
        # otimizadores dos pesos
        self.W_opt = copy.copy(optimizer)
        self.w0_opt = copy.copy(optimizer)
    
    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.w0.shape)
    
    def forwardPass(self, X, training):
        self.layer_input = X
        return X.dot(self.W) + self.w0
    
    def backwardPass(self, accumulated_gradient):
        # salvando os pesos usados durante o forward
        W = self.W

        if self.trainable:
            # calculando o gradiente dos pesos da camada w.r.t
            