from neuron import Neuron
import numpy as np
from activation_function import ActivationFunction


class Layer:
    """
    The Layer class holds a list of neurons of size 'size' assigning to them by default an activation function
    """

    def __init__(self, size, activation_fun):
        self.neurons_count = size
        self.neurons = [Neuron(activation_fun) for n in range(size)]
        self.activation_function = activation_fun

    def activate(self, inputs):
        """
        Activate the layer using the activation functions of the neurons
        :param inputs: numpy array matching the number of neurons in the layer
        :return: numpy array matching the number of neurons in the layer after activation function
        """
        if inputs.size == self.neurons_count:
            if self.neurons_count == 1:
                return np.array(self.neurons[0].fire(inputs))
            else:
                activated = []
                for n in range(self.neurons_count):
                    activated.append(self.neurons[n].fire(inputs[n]))
                return np.array(activated)
        else:
            raise Exception("Input and layer not same length")

    def __len__(self):
        return self.neurons_count
