class Neuron:
    """
    This class defines a neuron. A neuron:
    - takes data in input
    - has an activation function
    - gives output based on activation function

    """

    def __init__(self, activation_function):
        self.__activation_function = activation_function

    def fire(self, val):
        return self.__activation_function(val)
