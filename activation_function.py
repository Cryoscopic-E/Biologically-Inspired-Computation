from math import cos
from math import tanh
from math import exp


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
        return 1 / (1 + exp(-data))

    @staticmethod
    def hyperbolic_tangent(data):
        return tanh(data)

    @staticmethod
    def cosine(data):
        return cos(data)

    @staticmethod
    def gaussian(data):
        return exp(-((data ** 2) / 2))

    @staticmethod
    def identity(data):
        return data
