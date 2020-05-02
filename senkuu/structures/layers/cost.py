from . import Layer
from ...strategies import Losses
from ...strategies import Metrics


class Cost(Layer):

    def __init__(self, loss, metrics):
        super().__init__()
        self.verbose = 1

        # loss & metrics
        self.f = Losses().select(loss)  # cost function
        self.error = None
        self.metrics = [Metrics().select(metric) for metric in metrics] if metrics else []
        self.score = None

        # parameters
        self.a = None
        self.y = None
        self.dx = None

    def forepropagation(self):
        # check dimension
        if self.y.shape[0] != self.prevlayer.units:
            raise Exception('dimension of y is not fit for units of Output layer')

        # get data
        self.a = self.prevlayer.a

        # loss
        self.error = self.f(self.a, self.y).sum(axis=0, keepdims=True).mean()

        # metrics
        self.score = [metric(self.a, self.y) for metric in self.metrics]

    def backpropagation(self):
        self.dx = self.f(self.a, self.y, derivative=True)
