import numpy as np


class Optimizers(object):

    def __init__(self):
        self.optimizers = {
            'sgd': SGD, 'SGD': SGD,
            'momentum': Momentum,
            'rmsprop': RMSprop, 'RMSprop': RMSprop,
            'adam': Adam, 'Adam': Adam
        }

    def select(self, optimizer):
        return self.optimizers[optimizer]()


class Optimizer(object):

    def __init__(self, alpha=0.01, decay=0.0):
        self.alpha = alpha
        self.decay = decay

    def initialize(self, layer):
        pass

    def update(self, layer, epoch, **kw):
        pass


class SGD(Optimizer):

    def update(self, layer, epoch, **kw):
        alpha = 1 / (1 + self.decay * epoch) * self.alpha
        layer.w -= alpha * layer.dw
        layer.b -= alpha * layer.db


class Momentum(Optimizer):

    def __init__(self, alpha=0.001, decay=0.0, *, beta=0.9):
        super().__init__(alpha, decay)
        self.beta = beta

    def initialize(self, layer):
        layer.vdw = np.zeros_like(layer.w)
        layer.vdb = np.zeros_like(layer.b)

    def update(self, layer, epoch, **kw):
        _alpha = 1 / (1 + self.decay * epoch) * self.alpha
        layer.vdw = self.beta * layer.vdw + (1 - self.beta) * layer.dw
        layer.vdb = self.beta * layer.vdb + (1 - self.beta) * layer.db

        layer.w -= _alpha * layer.vdw
        layer.b -= _alpha * layer.vdb


class RMSprop(Optimizer):

    def __init__(self, alpha=0.001, decay=0.0, *, beta=0.999, epsilon=1e-08):
        super().__init__(alpha, decay)
        self.beta = beta
        self.epsilon = epsilon

    def initialize(self, layer):
        layer.sdw = np.zeros_like(layer.w)
        layer.sdb = np.zeros_like(layer.b)

    def update(self, layer, epoch, **kw):
        _alpha = 1 / (1 + self.decay * epoch) * self.alpha
        layer.sdw = self.beta * layer.sdw + (1 - self.beta) * (layer.dw ** 2)
        layer.sdb = self.beta * layer.sdb + (1 - self.beta) * (layer.db ** 2)

        layer.w -= _alpha * (layer.dw / (np.sqrt(layer.sdw) + self.epsilon))
        layer.b -= _alpha * (layer.db / (np.sqrt(layer.sdb) + self.epsilon))


class Adam(Optimizer):

    def __init__(self, alpha=0.001, decay=0.0, *, beta1=0.9, beta2=0.999, epsilon=1e-08):
        super().__init__(alpha, decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def initialize(self, layer):
        layer.vdw = np.zeros_like(layer.w)
        layer.vdb = np.zeros_like(layer.b)
        layer.sdw = np.zeros_like(layer.w)
        layer.sdb = np.zeros_like(layer.b)

    def update(self, layer, epoch, **kw):
        iteration = kw.get('iteration')

        _alpha = 1 / (1 + self.decay * epoch) * self.alpha
        layer.vdw = self.beta1 * layer.vdw + (1 - self.beta1) * layer.dw
        layer.vdb = self.beta1 * layer.vdb + (1 - self.beta1) * layer.db
        layer.sdw = self.beta2 * layer.sdw + (1 - self.beta2) * (layer.dw ** 2)
        layer.sdb = self.beta2 * layer.sdb + (1 - self.beta2) * (layer.db ** 2)

        vcdw = layer.vdw / (1 - self.beta1 ** iteration)
        vcdb = layer.vdb / (1 - self.beta1 ** iteration)
        scdw = layer.sdw / (1 - self.beta2 ** iteration)
        scdb = layer.sdb / (1 - self.beta2 ** iteration)

        layer.w -= _alpha * (vcdw / (np.sqrt(scdw) + self.epsilon))
        layer.b -= _alpha * (vcdb / (np.sqrt(scdb) + self.epsilon))
