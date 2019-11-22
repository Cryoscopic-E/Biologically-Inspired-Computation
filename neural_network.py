import numpy as np
from layer import Layer
from nn_sets import NNSets
from activation_function import ActivationFunction


class NeuralNetwork:

    def __init__(self, training_set):
        self.training_set = training_set
        out = []
        for t_set in self.training_set:
            out.append(t_set.output)
        self.expected_out = np.array(out)
        self.layers = []
        self.neurons = []
        self.total_n = 0

    def create_layer(self, n_neurons, act_fun):
        self.total_n += n_neurons
        layer = Layer(n_neurons, act_fun)
        self.layers.append(layer)
        self.neurons += layer.neurons

    # def feed_forward(self, inputs, desired_output, weights):
    #     """
    #     Perform the feed forward of the neural network, calculating the final output layer array or value
    #
    #     :param inputs: numpy array matching the number of neurons in the input layer
    #     :param desired_output: numpy array matching the number of neurons in the output layer
    #     :param weights: numpy array of all weights for the neural network (flat array)
    #     :return: mean squared error of the network
    #     """
    #     if len(self.layers) > 1:  # at least a perceptron
    #         start = 0
    #         current_l = self.layers[0].activate(inputs).reshape((1, -1)).T
    #         for n in range(1, len(self.layers)):
    #             n_weights = len(self.layers[n]) * current_l.size
    #             cut = start + n_weights
    #             _w = weights[start:cut]
    #             _w = _w.reshape((self.layers[n].get_size(), current_l.size))
    #             dot_prod = NeuralNetwork.dot_prod(current_l, _w)
    #             next_l = self.layers[n].activate(dot_prod)
    #             current_l = next_l
    #             start = n_weights
    #         print("Expected", desired_output, "Found", current_l)
    #         return NeuralNetwork.mse(current_l, desired_output)
    #     else:
    #         raise Exception("Invalid number of layers")

    def feed_forward(self, weights, _set):
        """
        Perform the feed forward of the neural network, calculating the final output layer array or value

        :param inputs: numpy array matching the number of neurons in the input layer
        :param desired_output: numpy array matching the number of neurons in the output layer
        :param weights: numpy array of all weights for the neural network (flat array)
        :return: mean squared error of the network
        """
        count = 0
        n_input = len(self.layers[0])
        act_f_w = weights[len(weights) - self.total_n:len(weights)]
        for neuron, af_weight in zip(self.neurons, act_f_w):
            if count < n_input:
                continue
            if -1.0 <= af_weight < -0.75:
                neuron.__activation_function = ActivationFunction.identity
            if -0.75 <= af_weight < -0.5:
                neuron.__activation_function = ActivationFunction.gaussian
            if -0.5 <= af_weight < -0.25:
                neuron.__activation_function = ActivationFunction.cosine
            if -0.25 <= af_weight < 0.0:
                neuron.__activation_function = ActivationFunction.hyperbolic_tangent
            if 0.0 <= af_weight < 0.25:
                neuron.__activation_function = ActivationFunction.sigmoid
            if 0.25 <= af_weight < 0.5:
                neuron.__activation_function = ActivationFunction.step
            if 0.5 <= af_weight < 0.75:
                neuron.__activation_function = ActivationFunction.soft_sign
            else:
                neuron.__activation_function = ActivationFunction.null
            count += 1
        if len(self.layers) > 1:  # at least a perceptron
            calc_out = []
            for t_set in _set:
                start = 0
                current_l = self.layers[0].activate(t_set.input).reshape((1, -1)).T

                for n in range(1, len(self.layers)):
                    n_weights = len(self.layers[n]) * current_l.size
                    cut = start + n_weights
                    _w = weights[start:cut]
                    _w = _w.reshape((len(self.layers[n]), current_l.size))
                    dot_prod = NeuralNetwork.dot_prod(current_l, _w)
                    next_l = self.layers[n].activate(dot_prod)
                    current_l = next_l
                    start = n_weights
                calc_out.append(current_l)
            return {"mse": self.mse(np.array(calc_out)),
                    "outputs": np.array(calc_out)}
        else:
            raise Exception("Invalid number of layers")

    def get_dimensions(self):
        dim = 0
        curr = len(self.layers[0])
        for n in range(1, len(self.layers)):
            dim += curr * len(self.layers[n])
            curr = len(self.layers[n])
        return dim

    # @staticmethod
    # def mse(output_observed, output_desired):
    #     return np.square(np.subtract(output_desired, output_observed)).mean()

    def mse(self, output_observed):
        return np.square(np.subtract(self.expected_out, output_observed)).mean()

    @staticmethod
    def dot_prod(x, weights):
        return np.dot(weights, x)
