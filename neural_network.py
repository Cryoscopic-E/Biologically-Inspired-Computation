import numpy as np
from neuron import Neuron
from layer import Layer
from activation_function import ActivationFunction


class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def create_layer(self, n_neurons, act_fun):
        self.layers.append(Layer(n_neurons, act_fun))

    def feed_forward(self, inputs, desired_output, weights):
        """
        Perform the feed forward of the neural network, calculating the final output layer array or value

        :param inputs: numpy array matching the number of neurons in the input layer
        :param desired_output: numpy array matching the number of neurons in the output layer
        :param weights: numpy array of all weights for the neural network (flat array)
        :return: mean squared error of the network
        """
        if len(self.layers) > 1:  # at least a perceptron
            start = 0
            current_l = self.layers[0].activate(inputs).reshape((1, -1)).T
            for n in range(1, len(self.layers)):
                n_weights = len(self.layers[n]) * current_l.size
                cut = start + n_weights
                _w = weights[start:cut]
                _w = _w.reshape((self.layers[n].get_size(), current_l.size))
                dot_prod = NeuralNetwork.dot_prod(current_l, _w)
                next_l = self.layers[n].activate(dot_prod)
                current_l = next_l
                start = n_weights
            print("Expected: ", desired_output, "Obtained: ", current_l)
            return NeuralNetwork.mse(current_l, desired_output)
        else:
            raise Exception("Invalid number of layers")

    def get_dimensions(self):
        dim = 0
        curr = self.layers[0].neurons
        for n in range(1, len(self.layers)):
            dim += curr * self.layers[n].neurons
            curr = self.layers[n].neurons
        return dim

    @staticmethod
    def mse(output_observed, output_desired):
        return np.square(np.subtract(output_desired, output_observed)).mean()

    @staticmethod
    def dot_prod(x, weights):
        return np.dot(weights, x)
