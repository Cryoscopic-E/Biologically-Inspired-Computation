import unittest
from neuron import Neuron
from activation_function import ActivationFunction

af = ActivationFunction()


class TestNeuronMethods(unittest.TestCase):

    def test_activation_function_null(self):
        null_expected_output = 0
        n = Neuron(af.null)
        self.assertEqual(n.activate([1, 2], [0.1, 0.2]), null_expected_output)

    def test_activation_function_sigmoid(self):
        sigmoid_expected_output = 0.6224593312018546
        n = Neuron(af.sigmoid)
        self.assertEqual(n.activate([1, 2], [0.1, 0.2]), sigmoid_expected_output)

    def test_activation_function_hyperbolic_tangent(self):
        hyperbolic_tangent_expected_output = 0.46211715726000974
        n = Neuron(af.hyperbolic_tangent)
        self.assertEqual(n.activate([1, 2], [0.1, 0.2]), hyperbolic_tangent_expected_output)

    def test_activation_function_cosine(self):
        cosine_expected_output = 0.8775825618903728
        n = Neuron(af.cosine)
        self.assertEqual(n.activate([1, 2], [0.1, 0.2]), cosine_expected_output)

    def test_activation_function_gaussian(self):
        gaussian_expected_output = 0.8824969025845955
        n = Neuron(af.gaussian)
        self.assertEqual(n.activate([1, 2], [0.1, 0.2]), gaussian_expected_output)


if __name__ == '__main__':
    unittest.main()
