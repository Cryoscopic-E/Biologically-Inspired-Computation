import numpy as np
from math import cos


class ActivationFunction:
    """
    Activation functions are created as methods. There are 5 types:
    Null, Sigmoid, Hyperbolic Tangent, Cosine, Gaussian
    """

    @staticmethod
    def null(data):
        return 0

    @staticmethod
    def sigmoid(data):
        return 1 / (1 + np.exp(-data))

    @staticmethod
    def hyperbolic_tangent(data):
        return np.tanh(data)

    @staticmethod
    def cosine(data):
        return cos(data)

    @staticmethod
    def gaussian(data):
        return np.exp(-((data ** 2) / 2))