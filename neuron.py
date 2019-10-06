class Neuron:
    """
    This class defines a neuron. A neuron:
    - takes data in input
    - has an activation function
    - gives output based on activation function

    """
    def __init__(self, activation_function):
        self.__activation_function = activation_function

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

####++++ try with "activation function" to test ++++####
# TODO: unittest for this class in other script + create folder Tests
# def tangent(sums):
#     return sums + 1
#
#
# def main():
#     neuron = Neuron(tangent)
#     neuron.activate([1, 2], [0.1, 0.2])
#
#
# if __name__ == "__main__":
#     main()
