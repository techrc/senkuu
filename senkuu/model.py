import numpy as np

from .structures.layers import Cost
from .strategies import Optimizers
from .strategies.optimizers import Optimizer
from .utils import split, to_class


class Model(object):

    def __init__(self):
        self.layers = []
        self.optimizer = None

        # logs
        self.losses = {'train': [], 'val': []}
        self.scores = {'train': [], 'val': []}

    # ----------------------------------------------------------------------------------------------------
    # configuration
    # ----------------------------------------------------------------------------------------------------

    def add(self, layer):
        """add layer into model"""

        # add
        self.layers.append(layer)

        # link
        if len(self.layers) > 1:
            self.layers[-1].prevlayer = self.layers[-2]
            self.layers[-2].nextlayer = self.layers[-1]

    def set(self, loss, metrics=None, optimizer='adam'):
        """configure some strategies such as loss, optimizer and metrics"""

        # loss
        if isinstance(loss, str):
            if isinstance(metrics, list) or metrics:  # metrics is list or None
                self.add(Cost(loss, metrics))
            else:
                raise TypeError('metrics should be a list or None')
        else:
            raise TypeError('loss should be a string')

        # optimizer
        if isinstance(optimizer, str):
            self.optimizer = Optimizers().select(optimizer)
        elif isinstance(optimizer, Optimizer):
            self.optimizer = optimizer
        else:
            raise TypeError('optimizer should be given Optimizer object or string')

    # ----------------------------------------------------------------------------------------------------
    # training / testing / predicting
    # ----------------------------------------------------------------------------------------------------

    def train(self, x, y, epochs=3000, size=64, validation=0.0, shuffle=False, verbose=1):
        """the whole processing of training"""

        if shuffle:
            np.random.shuffle(x)
            np.random.shuffle(y)

        # training
        iteration = 0  # counter
        batches = int(np.ceil(x.shape[0] / size))
        for epoch in range(epochs):
            for batch in range(batches):

                # processing data
                start, stop = batch * size, (batch+1) * size
                batch_x, batch_y = x[start:stop], y[start:stop]
                train_x, train_y, val_x, val_y = split(batch_x, batch_y, validation, shuffle=False)

                # train set
                self.__train(train_x, train_y, epoch, batch, iteration=iteration)
                self.__record('train')

                # validation set
                if validation:
                    self.test(val_x, val_y)
                    self.__record('val')

                # report result
                if verbose:
                    result = 'Epoch:{epoch:>4}/{epochs} - Batch:{batch:>2}/{batches}'.format(
                        epoch=epoch+1, epochs=epochs, batch=batch+1, batches=batches
                    )
                    print(result)
                    self.__report('train')
                    if validation:
                        self.__report('val')

                iteration += 1  # total iteration + 1

        if not verbose:  # verbose == 0
            self.__report('train')

    def __train(self, x, y, epoch, batch, **kw):
        """the core processing of training"""

        # feeding data
        self.__feed(x.T, y.T)

        # initialization or updating
        if epoch == batch == 0:
            for layer in self.layers[1:-1]:  # Input & Cost layers have no parameters to initialize
                layer.initialization()
                self.optimizer.initialize(layer)
        else:
            for layer in self.layers[1:-1]:  # Input & Cost layers have no parameters to update
                self.optimizer.update(layer, epoch, **kw)

        # fore propagation
        for layer in self.layers:
            layer.forepropagation()

        # back propagation
        for layer in self.layers[::-1]:
            layer.backpropagation()

    def test(self, x, y):
        """the processing of testing or validation"""

        self.__feed(x.T, y.T)
        for layer in self.layers:
            layer.forepropagation()

        return self.layers[-1].error, self.layers[-1].score

    def predict(self, x, onlyclass=True):
        """the processing of applying trained model to predict something"""

        self.__feed(x.T)
        for layer in self.layers[:-1]:  # no need Cost layer in predicting
            layer.forepropagation()
        y = self.layers[-2].a
        print('prediction:\n', to_class(y) if onlyclass else y)

    def __feed(self, x, y=None):
        """feed data set into layer"""

        self.layers[0].x = x
        self.layers[-1].y = y

    def __record(self, process):
        """record loss and scores in training or validation"""

        self.losses[process].append(self.layers[-1].error)
        self.scores[process].append(self.layers[-1].score)

    def __report(self, process):
        """print results of training or testing"""

        content = 'loss of {process:^5}: {loss:>.7f}   Metrics: {scores}'.format(
            process=process,
            loss=self.losses[process][-1],
            scores=self.scores[process][-1]
        )
        print(content)
