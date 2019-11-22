from matplotlib import pyplot as plt
from math import cos, sin, atan


class DrawNeuron():
    """
    Class to draw a single neuron
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def draw_neron(self, neuron_radius):
        circle = plt.Circle((self.x, self.y), radius=neuron_radius, fill=True)
        plt.gca().add_patch(circle)


class DrawLayer():
    """
    Class to draw a layer
    """
    def __init__(self, network, number_of_neurons, number_of_neurons_in_widest_layer):
        self.horizontal_distance_between_layers = 5
        self.vertical_distance_between_neurons = 2
        self.neuron_radius = 0.5
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.previous_layer = self.__get_previous_layer(network)
        self.x = self.__calculate_layer_x_position()
        self.neurons = self.__intialise_neurons(number_of_neurons)

    def __intialise_neurons(self, number_of_neurons):
        neurons = []
        y = self.__calculate_left_margin_so_layer_is_centered(number_of_neurons)
        for iteration in range(number_of_neurons):
            neuron = DrawNeuron(self.x, y)
            neurons.append(neuron)
            y += self.vertical_distance_between_neurons
        return neurons

    def __calculate_left_margin_so_layer_is_centered(self, number_of_neurons):
        return self.vertical_distance_between_neurons * (self.number_of_neurons_in_widest_layer - number_of_neurons) / 2

    def __calculate_layer_x_position(self):
        if self.previous_layer:
            return self.previous_layer.x - self.horizontal_distance_between_layers
        else:
            return 0

    def __get_previous_layer(self, network):
        if len(network.layers) > 0:
            return network.layers[-1]
        else:
            return None

    def __line_between_two_neurons(self, neuron1, neuron2):
        angle = atan(float(neuron2.y + neuron1.y) / (neuron2.x - neuron1.x))
        x_adjustment = self.neuron_radius * sin(angle)
        y_adjustment = self.neuron_radius * cos(angle)
        line = plt.Line2D((neuron1.x + x_adjustment, neuron2.x - x_adjustment),
                          (neuron1.y - y_adjustment, neuron2.y - y_adjustment))
        plt.gca().add_line(line)

    def draw_layer(self, layerType=0):
        for neuron in self.neurons:
            neuron.draw_neron(self.neuron_radius)
            if self.previous_layer:
                for previous_layer_neuron in self.previous_layer.neurons:
                    self.__line_between_two_neurons(neuron, previous_layer_neuron)
        # write Text
        x_text = self.number_of_neurons_in_widest_layer * self.vertical_distance_between_neurons
        if layerType == 0:
            plt.text(x_text, self.x, 'Input Layer', fontsize=10)
        elif layerType == -1:
            plt.text(x_text, self.x, 'Output Layer', fontsize=10)
        else:
            plt.text(x_text, self.x, 'Hidden Layer ' + str(layerType), fontsize=10)


class DrawNeuralNetwork():
    """
    Class to draw the Neural Network
    """
    def __init__(self, number_of_neurons_in_widest_layer):
        self.number_of_neurons_in_widest_layer = number_of_neurons_in_widest_layer
        self.layers = []
        # self.layertype = 0

    def add_layer(self, number_of_neurons):
        layer = DrawLayer(self, number_of_neurons, self.number_of_neurons_in_widest_layer)
        self.layers.append(layer)

    def draw_nn(self):
        fig = plt.figure()
        for i in range(len(self.layers)):
            layer = self.layers[i]
            if i == len(self.layers) - 1:
                i = -1
            layer.draw_layer(i)
        plt.axis('scaled')
        plt.axis('off')
        plt.title('Neural Network architecture', fontsize=15)
        plt.show()
        fig.savefig("ann.jpg")


class DrawANN():
    def __init__(self, neural_network):
        self.neural_network = neural_network

    def draw_ann(self):
        widest_layer = max(self.neural_network)
        network = DrawNeuralNetwork(widest_layer)
        for l in self.neural_network:
            network.add_layer(l)
        network.draw_nn()


# def main():
#     ann = DrawANN([2, 6, 8, 1])
#     ann.draw_ann()

# if __name__ == "__main__":
#     main()
