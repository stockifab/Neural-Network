import numpy as np


def get_goal(x, y):
    return y > -0.0001 * x**3 + 70


def generate(n=1000):
    data = np.empty((n, 2))
    target = np.empty((n, 1))

    for i in range(n):
        x = np.random.randint(0, 100)
        y = np.random.randint(0, 100)

        data[i] = (x, y)
        target[i] = (get_goal(x, y),)

    return data, target
