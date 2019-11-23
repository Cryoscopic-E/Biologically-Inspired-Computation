from neuron import Neuron
import numpy as np


class Layer:
    """
    The Layer class holds a list of neurons of size 'size' assigning to them by default an activation function
    """

    def __init__(self, size, activation_fun, _type):
        self.neurons_count = size
        self.neurons = [Neuron(activation_fun, _type) for n in range(size)]

    def activate(self, inputs):
        """
        Activate the layer using the activation functions of the neurons
        :param inputs: numpy array matching the number of neurons in the layer
        :return: numpy array matching the number of neurons in the layer after activation function
        """
        # activate if input vector same size of number of neurons in layer
        if inputs.size == self.neurons_count:
            # only one neuron not using a vector
            if self.neurons_count == 1:
                return np.array(self.neurons[0].fire(inputs))
            else:
                # more than one neuron/input
                activated = []
                for n in range(self.neurons_count):
                    # use neuron's activation function
                    r = self.neurons[n].fire(inputs[n])
                    activated.append(r)
                return np.array(activated)
        else:
            raise Exception("Input and layer not same length")

    def __len__(self):
        return self.neurons_count
