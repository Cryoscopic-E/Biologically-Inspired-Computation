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

    '''
    matrix is calculated for each layer by taking in input the numbers of neurons in the current layer
    and the numbers of neurons in the following layer.
    e.g. if I need the weights for the first layer (input layer), this method will
    create a weight matrix from input to hidden layer by taking the numbers of neuron in input and 
    the numbers of neurons in the hidden layer.
    '''
    @staticmethod
    def random_weights(n_neurons_layer, n_neurons_following_layer):
        return np.random.randn(n_neurons_layer, n_neurons_following_layer)

    # Mean Squared Error (MSE)
    @staticmethod
    def mse(output_observed, output_desired):
        return np.square(np.subtract(output_desired, output_observed)).mean()
