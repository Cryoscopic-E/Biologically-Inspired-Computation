from neuron import Neuron
import numpy as np


class Layer:
    """
    The Layer class holds a list of neurons of size 'size' assigning to them by default an activation function
    """

    def __init__(self, size, activation_fun):
        self.neurons = size
        self.activation_function = activation_fun

    def fire_af(self, val):
        return self.activation_function(val)

    def activate(self, inputs):
        """
        Activate the layer using the activation functions of the neurons
        :param inputs: numpy array matching the number of neurons in the layer
        :return: numpy array matching the number of neurons in the layer after activation function
        """
        if inputs.size == self.neurons:
            # return np.array(list(map(self.fire_af, inputs)))
            return np.vectorize(self.fire_af)(inputs)
        else:
            raise Exception("Input and layer not same length")

    def get_size(self):
        return self.neurons

    def __len__(self):
        return self.neurons
