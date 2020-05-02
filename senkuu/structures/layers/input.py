from .base import Layer


class Input(Layer):

    def __init__(self, units):
        super().__init__()

        self.units = units

        self.x = None
        self.a = None

    def forepropagation(self):
        # check dimension
        if self.x.shape[0] != self.units:
            raise Exception('dimension of x is not fit for units of Input layer')

        # get data
        self.a = self.x
