import numpy as np


# loss function factory
class Losses(object):

    def __init__(self):
        self.losses = {
            'mse': mse, 'MSE': mse,

            'binary': binary_cross_entropy,
            'binary_crossentropy': binary_cross_entropy,

            'categorical': categorical_cross_entropy,
            'categorical_crossentropy': categorical_cross_entropy
        }

    def select(self, loss):
        return self.losses[loss]


def mse(a, y, derivative=False):
    if not derivative:
        return 0.5 * (y - a) ** 2
    else:
        return -(y - a)


def binary_cross_entropy(a, y, derivative=False):
    if not derivative:
        return -(y * np.log(a + 1e-08) + (1 - y) * np.log(1 - a + 1e-08))
    else:
        return (a - y) / (a * (1 - a) + 1e-08)


def categorical_cross_entropy(a, y, derivative=False):
    if not derivative:
        return -y * np.log(a + 1e-08)
    else:
        return - y / (a + 1e-08)
