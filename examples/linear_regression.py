import numpy as np

from senkuu.model import Model
from senkuu.structures.layers import Input, Dense


np.random.seed(2020)

x = np.linspace(-5, 5, 128).reshape(-1, 1)
y = 2 * x

model = Model()
model.add(Input(units=1))
model.add(Dense(units=1, activation='linear'))
model.set(loss='mse', optimizer='sgd')

model.train(x, y, epochs=100)
