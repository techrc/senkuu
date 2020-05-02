import numpy as np
from . import Layer
from ...strategies import Activations


class Dense(Layer):

    def __init__(self, units, activation):
        super().__init__()
        self.units = units
        self.activation = activation
        self.g = Activations().select(activation)

        # fore parameters
        self.x = None
        self.w = None
        self.b = None
        self.z = None
        self.a = None

        # back parameters
        self.da = None
        self.dw = None
        self.db = None
        self.dx = None

    def forepropagation(self):
        # get data
        self.x = self.prevlayer.a

        self.z = self.w @ self.x + self.b
        self.a = self.g(self.z)

    def backpropagation(self):
        # get data
        self.da = self.nextlayer.dx

        dadz = self.g(self.z, derivative=True)
        if self.activation == 'softmax':
            n, m = self.da.shape[0], self.da.shape[1]
            dz = (dadz @ self.da.reshape(m, n, 1)).reshape(m, n).T
        else:
            dz = self.da * dadz

        self.dw = dz @ self.x.T / self.x.shape[1]
        self.db = dz.mean(axis=1, keepdims=True)
        self.dx = self.w.T @ dz

    def initialization(self):
        self.w = np.random.randn(self.units, self.prevlayer.units) * 0.01
        self.b = np.zeros([self.units, 1])
