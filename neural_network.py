import numpy as np
from layer import Layer
from nn_sets import NNSets
from activation_function import ActivationFunction


class NeuralNetwork:

    def __init__(self, training_set):
        self.training_set = training_set
        self.layers = []
        self.neurons = []
        self.total_n = 0

    def create_layer(self, n_neurons, act_fun, _type):
        """
        Create a new layer and append to the layers array of the neural network

        :param n_neurons: total number of neuron in a layer
        :param act_fun: default activation function for the neurons
        :param _type: type of neurons (input, hidden, output)
        """
        self.total_n += n_neurons
        layer = Layer(n_neurons, act_fun, _type)
        self.layers.append(layer)
        self.neurons += layer.neurons

    def feed_forward(self, weights_positions, af_positions, bias_positions, _set):
        """
        Perform the feed forward of the neural network, calculating the final output layer array or value

        :param weights_positions: numpy array of all weights positions
        :param af_positions: numpy array of all activation functions positions
        :param bias_positions: numpy array of all bias positions
        :param _set: set on which perform the feed forward (inputs, expected output)
        :return: mean squared error of the network
        """
        # Setup each neuron using particle's respective positions
        self._setup_neurons(af_positions, bias_positions)

        # initialize the output list as empty
        outputs_list = []
        expected_out = []
        for t_set in _set:
            # starting index used to slice the positions' weights array
            start_index = 0
            # starting layer set to input layer in form [[x0],[x1],...,[xn]] (transposed)
            current_layer_input = self.layers[0].activate(t_set.input.copy()).reshape((1, -1)).T
            # loop from first hidden layer
            for n in range(1, len(self.layers)):
                # calculate total number of weights in connection current_layer->following_layer
                total_weights_current_x_following = len(self.layers[n]) * current_layer_input.size
                # slice index from start to number of weights
                slice_index = start_index + total_weights_current_x_following
                # slice the positions weights array and reshape to fit the weights matrix of current layer (for dot product)
                weights_matrix = weights_positions[start_index:slice_index].reshape((len(self.layers[n]), current_layer_input.size))
                # perform dot product ( deep explanation in report)
                dot_prod = NeuralNetwork.dot_prod(current_layer_input, weights_matrix)
                # activate following layer using the dot product vector calculated
                next_layer_output = self.layers[n].activate(dot_prod)
                # new input is previous output
                current_layer_input = next_layer_output
                # update start index to slice next weights
                start_index = total_weights_current_x_following
            # after a feed forward add the output to the list
            outputs_list.append(current_layer_input)
            expected_out.append(t_set.output)
        # return results
        return {"mse": NeuralNetwork.mse(np.array(outputs_list, copy=True), np.array(expected_out, copy=True)),
                "outputs": np.array(outputs_list)}

    def get_total_weights(self):
        """
        Calculate total number of weights
        Multiply following layer and previous layer adding for all layers in neural network
        
        :return: total number of weights in the neural network
        """
        n_weights = 0
        curr = len(self.layers[0])
        for n in range(1, len(self.layers)):
            n_weights += curr * len(self.layers[n])
            curr = len(self.layers[n])
        return n_weights

    def _setup_neurons(self, af_positions, bias_positions):
        """
        Set up the neurons of the neural network using particle's positions

        :param af_positions: positions of activation functions
        :param bias_positions: positions of bias value
        """
        for neuron, af, bias in zip(self.neurons, af_positions, bias_positions):
            # Set bias value
            neuron.bias = bias

            # Set activation function
            # Skip the input neuron (fixed to identity af)
            if neuron.type == 'input':
                neuron.activation_function = ActivationFunction.identity
                continue
            else:
                # Select the appropriate activation function
                if -1.0 <= af < -0.75:
                    neuron.activation_function = ActivationFunction.identity
                elif -0.75 <= af < -0.5:
                    neuron.activation_function = ActivationFunction.gaussian
                elif -0.5 <= af < -0.25:
                    neuron.activation_function = ActivationFunction.cosine
                elif -0.25 <= af < 0.0:
                    neuron.activation_function = ActivationFunction.hyperbolic_tangent
                elif 0.0 <= af < 0.25:
                    neuron.activation_function = ActivationFunction.sigmoid
                elif 0.25 <= af < 0.5:
                    neuron.activation_function = ActivationFunction.step
                elif 0.5 <= af < 0.75:
                    neuron.activation_function = ActivationFunction.soft_sign
                else:
                    neuron.activation_function = ActivationFunction.null
        pass

    @staticmethod
    def mse(output_observed, output_expected):
        """
        Calculate the mean squared error

        :param output_observed: list of output from neural network feed forward
        :param output_expected: list of output from the provided set of training/test
        :return: MSE value
        """
        return np.square(np.subtract(output_expected, output_observed)).mean()

    @staticmethod
    def dot_prod(x, weights):
        """
        Perform the dot product between inputs and respective weight matrix that connects them to the following layer

        :param x: input vector
        :param weights: weight matrix (following_layer,input)
        :return: vector or number
        """
        return np.dot(weights, x)
