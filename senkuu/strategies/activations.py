import numpy as np


class Activations(object):

    def __init__(self):
        self.activations = {
            'linear': linear,
            'sigmoid': sigmoid,
            'tanh': tanh,
            'relu': relu,
            'leaky_relu': leaky_relu,
            'softmax': softmax
        }

    def select(self, activation):
        return self.activations[activation]


def linear(z, derivative=False):
    if not derivative:
        return z
    else:
        return 1


def sigmoid(z, derivative=False):
    if not derivative:
        return 1 / (1 + np.exp(-z))
    else:
        return sigmoid(z) * (1 - sigmoid(z))


def tanh(z, derivative=False):
    if not derivative:
        return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))
    else:
        return 1 - tanh(z) ** 2


def relu(x, derivative=False):
    if not derivative:
        return np.max(0, x)
    else:
        return 1 if x > 0 else 0


def leaky_relu(x, derivative=False):
    if not derivative:
        return np.max(0.01*x, x)
    else:
        return 1 if x > 0 else 0.01


def softmax(x, derivative=False):
    if not derivative:
        t = np.exp(x)
        return t / t.sum(0)
    else:
        a = softmax(x)
        n, m = x.shape[0], x.shape[1]
        result = np.empty((m, n, n))

        for index in range(m):
            for x in range(n):
                for y in range(n):
                    if x == y:
                        result[index, x, y] = a[x, index] * (1 - a[x, index])
                    else:
                        result[index, x, y] = -a[x, index] * a[y, index]
        return result
