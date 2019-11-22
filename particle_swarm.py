from random import uniform, choice
from neural_network import NeuralNetwork
from activation_function import ActivationFunction
from nn_sets import NNSets
import numpy as np
import matplotlib.pyplot as plt
from progress.bar import Bar


class _Particle:
    pos_high_bound = 1.0
    pos_low_bound = -1.0

    def __init__(self, dimensions):
        self.positions = np.random.uniform(low=-1, high=1, size=dimensions)
        self._velocity = np.random.uniform(low=-0.1, high=0.1, size=dimensions)
        self.best_positions = np.random.uniform(low=-1, high=1, size=dimensions)
        self.fit = 100
        self.informant = None
        pass

    def update_positions(self):
        """
        Update particle's position using the velocity vector
        """
        for n in range(len(self.positions)):
            self.positions[n] += self._velocity[n]
            if self.positions[n] < _Particle.pos_low_bound or self.positions[n] > _Particle.pos_high_bound:
                self.positions[n] = uniform(_Particle.pos_low_bound, _Particle.pos_high_bound)
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
                              (beta * (self.best_positions[dim] - self.positions[dim])) + \
                              (gamma * (self.informant.best_positions[dim] - self.positions[dim])) + \
                              (delta * (global_best[dim] - self.positions[dim]))
        pass

    def set_best_pos(self):
        self.best_positions = self.positions
        pass

    def set_informant(self, particle):
        self.informant = particle
        pass


class PSO:
    global_best_positions = None
    global_best_fitness = 1

    def __init__(self, _sets, epochs, neural_network, swarm_size, dimensions, alpha, beta, gamma, delta):
        PSO.global_best_positions = np.random.uniform(low=-1, high=1, size=dimensions)
        self.epochs = epochs
        self.train_sets = _sets.training_set
        self.test_sets = _sets.test_set
        self.neural_network = neural_network
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

    def fitness(self, weights, _set):
        """
        Perform a feed forward of the neural network using the n particle positions as weights
        :param inputs:
        :param weights:
        :param outputs:
        :return: fitness value (squared mean error of nn)
        """
        return self.neural_network.feed_forward(weights, _set)["mse"]

    def fit(self):
        # for n in range(50):
        k = 0

        # PLOT INIT
        ###############################################
        x = []
        y = []
        ###############################################
        bar = Bar('Epoch ', max=self.epochs)
        for n in range(self.epochs):
            for particle in self.particles:
                # particle_fitness = self.fitness(train.input, train.output, particle.get_positions())
                # particle_best_fitness = self.fitness(train.input, train.output, particle.best_pos)
                #
                # if particle_fitness < particle_best_fitness:
                #     particle.fit = particle_fitness
                #     particle.set_best_pos()
                #
                # if self.global_best_positions is None:
                #     self.global_best_positions = particle.get_positions()
                # global_fitness = self.fitness(train.input, train.output, self.global_best_positions)
                #
                # if particle_best_fitness < self.global_best_fitness:
                #     self.global_best_fitness = global_fitness
                #     self.global_best_positions = particle.best_pos
                #     print("X best pos", self.global_best_positions, "of particle", particle)

                particle_fitness = self.fitness(particle.positions, self.train_sets)
                particle_best_fitness = self.fitness(particle.best_positions, self.train_sets)

                if particle_fitness < particle_best_fitness:
                    particle.fit = particle_best_fitness
                    particle.best_positions = particle.positions.copy()

                if particle_best_fitness < PSO.global_best_fitness:
                    PSO.global_best_fitness = particle_best_fitness
                    PSO.global_best_positions = particle.best_positions.copy()

                # particle_fitness = self.fitness(particle.positions, self.train_sets)
                # global_best_fit = self.fitness(PSO.global_best_positions, self.train_sets)
                # if particle_fitness < global_best_fit:
                #     PSO.global_best_fitness = particle_fitness
                #     PSO.global_best_positions = particle.positions.copy()

            if PSO.global_best_fitness < 0.1:
                break
            # PSO.global_best_positions = self._get_best_positions()

            # test_fitness = self.fitness(train.input, train.output, PSO.global_best_positions)
            # if test_fitness < PSO.global_best_fitness:
            #     PSO.global_best_fitness = test_fitness

            # if self.global_best_fitness < 0.01:
            #     break
            # PLOT GLOBAL BEST FITNESS
            ################################################
            k += 1
            x.append(k)
            y.append(PSO.global_best_fitness)
            ################################################

            for particle in self.particles:
                for dim in range(self.dimensions):
                    b = uniform(0, self.beta)
                    c = uniform(0, self.gamma)
                    d = uniform(0, self.delta)
                    particle.update_velocity(dim, self.alpha, b, c, d, PSO.global_best_positions)

                particle.update_positions()
            bar.next()
        bar.finish()
        plt.plot(x, y)
        plt.show()

    def _get_best_positions(self):
        best = self.particles[0]
        for particle in self.particles:
            if particle.fit < best.fit:
                best = particle
        return best.best_positions

    def predict(self):
        ff = self.neural_network.feed_forward(PSO.global_best_positions, self.test_sets)
        outs = ff["outputs"].ravel()
        mse = ff["mse"]
        ins = []
        exp = []
        for s in self.test_sets:
            ins.append(s.input)
            exp.append(s.output)

        for n in range(len(outs)):
            print("Input", ins[n], "Output", outs[n], "Expected", exp[n])
            print(mse)
        plt.scatter(ins, outs)
        plt.scatter(ins, exp)
        plt.show()
        pass


if __name__ == "__main__":
    sets = NNSets("./Data/1in_linear.txt")
    nn = NeuralNetwork(sets.training_set)
    nn.create_layer(1, ActivationFunction.identity)
    nn.create_layer(3, ActivationFunction.sigmoid)
    nn.create_layer(2, ActivationFunction.gaussian)
    nn.create_layer(1, ActivationFunction.identity)
    pso = PSO(sets, 200, nn, 40, nn.get_dimensions() + nn.total_n, .8, .5, 1.5, 2)
    pso.fit()
    print("Epochs", pso.epochs)
    print("Best fitness val", pso.global_best_fitness)
    print("best pos", pso.global_best_positions)
    pso.predict()
