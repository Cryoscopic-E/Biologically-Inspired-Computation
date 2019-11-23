from random import uniform


class Neuron:
    """
    This class defines a neuron. A neuron:
    - takes data in input
    - has an activation function
    - gives output based on activation function

    """

    def __init__(self, activation_function, _type):
        """
        Neuron constructor
        Randomly set the bias on startup
        :param activation_function: activation function to use
        """
        self.activation_function = activation_function
        self.bias = uniform(-1, 1)
        self.type = _type

    def fire(self, val):
        """
        Activate the neuron using the activation function
        :param val: result of dot product of each connected neuron
        :return:
        """
        if self.type == 'hidden':
            return self.activation_function(val + self.bias)
        return self.activation_function(val)
