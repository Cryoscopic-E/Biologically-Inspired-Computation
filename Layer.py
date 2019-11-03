from neuron import Neuron


class Layer:
    """
    The Layer class holds a list of neurons of size 'size' assigning to them by default an activation function
    """

    def __init__(self, size, activation_fun):
        self.neurons = [Neuron(activation_fun) for v in range(size)]

    def change_activation_function(self, index, new_function):
        if index > len(self.neurons) or not (index < 0):
            self.neurons[index].__activation_function = new_function
