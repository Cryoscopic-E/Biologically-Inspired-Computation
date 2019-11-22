import numpy as np
from math import cos


class ActivationFunction:
    """
    Activation functions are created as methods. There are 5 types:
    Null, Sigmoid, Hyperbolic Tangent, Cosine, Gaussian
    """

    @staticmethod
    def soft_sign(data):
        return data / (1 + abs(data))

    @staticmethod
    def step(data):
        """Logistic or Binary step function [0,1]"""
        return 0 if data < 0 else 1

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

    @staticmethod
    def identity(data):
        return data
