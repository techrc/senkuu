import numpy as np


def split(x, y, rate, shuffle=False):
    # shuffle
    data = np.hstack([x, y])
    if shuffle:
        np.random.shuffle(data)
    x, y = np.split(data, [x.shape[1]], axis=1)

    # splitting
    boundary = data.shape[0] - int(np.ceil(data.shape[0] * rate))
    train_x, test_x = np.split(x, [boundary])
    train_y, test_y = np.split(y, [boundary])

    return train_x, train_y, test_x, test_y


def to_class(a, boundary=0.5):
    return np.where(a >= boundary, 1, 0)
