from random import uniform, choice
from neural_network import NeuralNetwork
from activation_function import ActivationFunction
from nn_sets import NNSets
import numpy as np


class _Particle:
    pos_high_bound = 1.5
    pos_low_bound = -1.5

    def __init__(self, dimensions):
        self._positions = np.random.uniform(low=-1, high=1, size=dimensions)
        self._velocity = np.random.uniform(low=-0.05, high=0.05, size=dimensions)
        self.best_pos = self._positions
        self.fit = 0
        self.informant = None
        pass

    def update_positions(self):
        """
        Update particle's position using the velocity vector
        """
        for n in range(len(self._positions)):
            self._positions[n] += self._velocity[n]
            if self._positions[n] < _Particle.pos_low_bound or self._positions[n] > _Particle.pos_high_bound:
                self._positions[n] = uniform(_Particle.pos_low_bound, _Particle.pos_high_bound)
        pass

    def update_velocity(self, dim, alpha, beta, gamma, delta, global_best):
        """
        Update the velocity of a single particle, using weight for components

        :param dim: dimension index
        :param alpha: old velocity to retain
        :param beta: personal best position to retain
        :param gamma: informant's best position to retain
        :param delta: global best's position to retain
        :param global_best: global best positions list
        """
        self._velocity[dim] = (alpha * self._velocity[dim]) + \
                              (beta * (self.best_pos[dim] - self._positions[dim])) + \
                              (gamma * (self.informant.best_pos[dim] - self._positions[dim])) + \
                              (delta * (global_best[dim] - self._positions[dim]))
        pass

    def get_positions(self):
        return self._positions

    def set_best_pos(self):
        self.best_pos = self._positions
        pass

    def set_informant(self, particle):
        self.informant = particle
        pass


class PSO:
    def __init__(self, data_file, neural_network, swarm_size, dimensions, alpha, beta, gamma, delta):
        self.all_sets = NNSets(data_file)
        self.neural_network = neural_network
        self.global_best_positions = None
        self.dimensions = dimensions
        self.particles = [_Particle(dimensions) for n in range(swarm_size)]
        self.init_informants_rnd()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        pass

    def init_informants_rnd(self):
        """
        Randomly pick a informant implementation
        :return:
        """
        for particle in self.particles:
            informant = choice(self.particles)
            while informant == particle:
                informant = choice(self.particles)

            particle.set_informant(informant)
        pass

    def fitness(self, inputs, outputs, weights):
        """
        Perform a feed forward of the neural network using the n particle positions as weights
        :param inputs:
        :param weights:
        :param outputs:
        :return: fitness value (squared mean error of nn)
        """
        return self.neural_network.feed_forward(inputs, outputs, weights)

    def fit(self):
        for n in range(30):
            for train in self.all_sets.get_training_set():
                for particle in self.particles:
                    particle_fitness = self.fitness(train.input, train.output, particle.get_positions())

                    if self.global_best_positions is None:
                        self.global_best_positions = particle.get_positions()

                    global_fitness = self.fitness(train.input, train.output, self.global_best_positions)

                    if particle_fitness < global_fitness:
                        particle.set_best_pos()
                        self.global_best_positions = particle.get_positions()

                for particle in self.particles:
                    for dim in range(self.dimensions):
                        b = uniform(0, self.beta)
                        c = uniform(0, self.gamma)
                        d = uniform(0, self.delta)
                        particle.update_velocity(dim, self.alpha, b, c, d, self.global_best_positions)

                    particle.update_positions()
        pass

    def predict(self):
        pass


if __name__ == "__main__":
    nn = NeuralNetwork()
    nn.create_layer(2, ActivationFunction.identity)
    nn.create_layer(1, ActivationFunction.sigmoid)
    nn.create_layer(1, ActivationFunction.hyperbolic_tangent)
    pso = PSO("./Data/2in_xor.txt", nn, 50, nn.get_dimensions(), 0.7, 1, 1, 2)
    pso.fit()
    print(nn.feed_forward(np.array([0., 0.]), np.array([0.]), pso.global_best_positions))
