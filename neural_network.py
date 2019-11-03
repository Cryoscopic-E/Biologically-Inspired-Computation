import numpy as np
from neuron import Neuron
from activation_function import ActivationFunction


class NeuralNetwork:

    def __init__(self, *args):
        self.input_layer = self._create_layer(args[0], ActivationFunction.null(0))
        self.output_layer = self._create_layer(args[len(args) - 1], ActivationFunction.null(0))
        pass

    @staticmethod
    def _create_layer(self, n_neurons, act_fun):
        return [Neuron(act_fun) for n in range(n_neurons)]
