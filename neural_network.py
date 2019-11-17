import numpy as np
from neuron import Neuron
from activation_function import ActivationFunction


class NeuralNetwork:

    def __init__(self, *args):
        self.input_layer = self._create_layer(args[0], ActivationFunction.identity(0))
        self.hidden_layer = self._create_layer(args[1], ActivationFunction.identity(1))
        self.output_layer = self._create_layer(args[len(args) - 1], ActivationFunction.null(0))
        self.input_to_hidden_weights = self._get_random_weights(len(self.input_layer), len(self.hidden_layer))
        self.hidden_to_output_weights = self._get_random_weights(len(self.hidden_layer), len(self.output_layer))

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
    def _get_random_weights(n_neurons_layer, n_neurons_following_layer):
        return np.random.randn(n_neurons_layer, n_neurons_following_layer)

    # Mean Squared Error (MSE)
    @staticmethod
    def mse(output_observed, output_desired):
        return np.square(np.subtract(output_desired, output_observed)).mean()

    @staticmethod
    def activate(self, inputs_vector, weights_vector):
        out = []
        sums_old = 0
        for input, weight in zip(inputs_vector, weights_vector):
            sums = input * weight
            sums_old = sums_old + sums
        output = self.__activation_function(sums_old)
        out.append(output)

        print(output)
        return output

    @staticmethod
    def activate_input_layer(self, inputs_vector):
        return inputs_vector

    @staticmethod
    def feedforward(self):
        first = self.activate(self.input_layer, self.input_to_hidden_weights)
        second = self.activate(first, self.hidden_to_output_weights)
        return first, second



