class Neuron:
    """
    This class defines a neuron. A neuron:
    - takes data in input
    - has an activation function
    - gives output based on activation function

    """

    def __init__(self, activation_function):
        self.__activation_function = activation_function

    def activate_input_layer(self, inputs_vector):
        return inputs_vector

    def activate(self, inputs_vector, weights_vector):
        out = []
        sums_old = 0
        for input, weight in zip(inputs_vector, weights_vector):
            sums = input * weight
            sums_old = sums_old + sums
        output = self.__activation_function(sums_old)
        out.append(output)

        return output
