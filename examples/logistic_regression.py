import numpy as np

from senkuu.model import Model
from senkuu.structures.layers import Input, Dense


np.random.seed(2020)

x = np.array([[-3], [-2], [-1], [1], [2], [3]])
y = np.array([[0], [0], [0], [1], [1], [1]])

model = Model()
model.add(Input(1))
model.add(Dense(1, 'sigmoid'))
model.set(loss='binary_crossentropy', metrics=['acc', 'precision', 'recall', 'f1'],
          optimizer='adam')

model.train(x, y, epochs=300, validation=0.2)

loss, score = model.test(x, y)
print(loss, score)

model.predict(x)
