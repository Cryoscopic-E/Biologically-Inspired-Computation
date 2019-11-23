from random import uniform, choice, sample
from neural_network import NeuralNetwork
from activation_function import ActivationFunction
from nn_sets import NNSets
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from progress.bar import Bar


class _Particle:
    pos_high_bound = 1.0
    pos_low_bound = -1.0
    max_velocity_mod = 0.1

    def __init__(self, n_weights, n_neurons):
        self.positions_weights = np.random.uniform(low=_Particle.pos_low_bound, high=_Particle.pos_high_bound, size=n_weights)
        self.positions_af = np.random.uniform(low=_Particle.pos_low_bound, high=_Particle.pos_high_bound, size=n_neurons)
        self.positions_bias = np.random.uniform(low=_Particle.pos_low_bound, high=_Particle.pos_high_bound, size=n_neurons)
        self._velocity_weights = np.random.uniform(low=-_Particle.max_velocity_mod, high=_Particle.max_velocity_mod, size=n_weights)
        self._velocity_af = np.random.uniform(low=-_Particle.max_velocity_mod, high=_Particle.max_velocity_mod, size=n_neurons)
        self._velocity_bias = np.random.uniform(low=-_Particle.max_velocity_mod, high=_Particle.max_velocity_mod, size=n_neurons)
        self.best_positions_weights = np.random.uniform(low=_Particle.pos_low_bound, high=_Particle.pos_high_bound, size=n_weights)
        self.best_positions_af = np.random.uniform(low=_Particle.pos_low_bound, high=_Particle.pos_high_bound, size=n_neurons)
        self.best_positions_bias = np.random.uniform(low=_Particle.pos_low_bound, high=_Particle.pos_high_bound, size=n_neurons)
        self.fit = 10
        self.informant = None
        pass

    def update_positions(self):
        """
        Update particle's positions using the respective velocity vector
        Epsilon always 1
        After the position is calculated checks if it's out of boundaries, if so it reposition itself randomly
        """
        for n in range(len(self.positions_weights)):
            self.positions_weights[n] += self._velocity_weights[n]
            if self.positions_weights[n] < _Particle.pos_low_bound or self.positions_weights[n] > _Particle.pos_high_bound:
                self.positions_weights[n] = uniform(_Particle.pos_low_bound, _Particle.pos_high_bound)

        for n in range(len(self.positions_af)):
            self.positions_af[n] += self._velocity_af[n]
            if self.positions_af[n] < _Particle.pos_low_bound or self.positions_af[n] > _Particle.pos_high_bound:
                self.positions_af[n] = uniform(_Particle.pos_low_bound, _Particle.pos_high_bound)

        for n in range(len(self.positions_bias)):
            self.positions_bias[n] += self._velocity_bias[n]
            if self.positions_bias[n] < _Particle.pos_low_bound or self.positions_bias[n] > _Particle.pos_high_bound:
                self.positions_bias[n] = uniform(_Particle.pos_low_bound, _Particle.pos_high_bound)
        pass

    def update_velocities(self, alpha, beta, gamma, delta, global_best_particle):
        """
        Update the velocity of a single particle, using the respective weights as components
        After velocity calculated, it's clamped on the maximum velocity permitted

        :param alpha: old velocity to retain
        :param beta: personal best position to retain
        :param gamma: informant's best position to retain
        :param delta: global best's position to retain
        :param global_best_particle: global best particle so far
        """
        for n in range(len(self._velocity_weights)):
            b = uniform(0, beta)
            c = uniform(0, gamma)
            d = uniform(0, delta)
            self._velocity_weights[n] = (alpha * self._velocity_weights[n]) + \
                                        (b * (self.best_positions_weights[n] - self.positions_weights[n])) + \
                                        (c * (self.informant.best_positions_weights[n] - self.positions_weights[n])) + \
                                        (d * (global_best_particle.best_positions_weights[n] - self.positions_weights[n]))
            if abs(self._velocity_weights[n] >= _Particle.max_velocity_mod):
                self._velocity_weights[n] = np.sign(self._velocity_weights[n]) * _Particle.max_velocity_mod
        for n in range(len(self._velocity_af)):
            b = uniform(0, beta)
            c = uniform(0, gamma)
            d = uniform(0, delta)
            self._velocity_af[n] = (alpha * self._velocity_af[n]) + \
                                   (b * (self.best_positions_af[n] - self.positions_af[n])) + \
                                   (c * (self.informant.best_positions_af[n] - self.positions_af[n])) + \
                                   (d * (global_best_particle.best_positions_af[n] - self.positions_af[n]))
            if abs(self._velocity_af[n] >= _Particle.max_velocity_mod):
                self._velocity_af[n] = np.sign(self._velocity_af[n]) * _Particle.max_velocity_mod
        for n in range(len(self._velocity_bias)):
            b = uniform(0, beta)
            c = uniform(0, gamma)
            d = uniform(0, delta)
            self._velocity_bias[n] = (alpha * self._velocity_bias[n]) + \
                                     (b * (self.best_positions_bias[n] - self.positions_bias[n])) + \
                                     (c * (self.informant.best_positions_bias[n] - self.positions_bias[n])) + \
                                     (d * (global_best_particle.best_positions_bias[n] - self.positions_bias[n]))
            if abs(self._velocity_bias[n] >= _Particle.max_velocity_mod):
                self._velocity_bias[n] = np.sign(self._velocity_bias[n]) * _Particle.max_velocity_mod
        pass

    def set_informant(self, particle_group):
        """
        Select the best particle in the group as informant, using also itself as informant

        :param particle_group: particle group randomly picked every iteration
        """
        best = self
        for particle in particle_group:
            if particle.fit < best.fit:
                best = particle
        self.informant = best
        pass


class PSO:
    """
    Implementation of the Particle Swarm Optimization class
    """

    def __init__(self, _sets, epochs, neural_network, swarm_size, alpha, beta, gamma, delta):
        """
        Basic PSO constructor

        :param _sets: training and test set reference
        :param epochs: max number of epochs
        :param neural_network: reference to neural network model to use
        :param swarm_size: max number of particles
        :param alpha: proportion of velocity to keep
        :param beta: proportion of best positions to keep
        :param gamma: proportions of informant's best positions to keep
        :param delta: proportions of global best particle's positions to keep
        """
        self.particles = [_Particle(neural_network.get_total_weights(), neural_network.total_n) for n in range(swarm_size)]
        self.global_best_particle = None
        self.global_best_fit = 10
        self.epochs = epochs
        self.train_sets = _sets.training_set
        self.test_sets = _sets.test_set
        self.neural_network = neural_network
        self.init_informants_rnd()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        pass

    def init_informants_rnd(self):
        """
        Randomly pick a informant implementation
        As initialization step an informant is randomly assigned to a particle (including itself)
        """
        for particle in self.particles:
            particle.informant = choice(self.particles)
        pass

    def fitness(self, weights, af, bias, _set):
        """
        Fitness calculation.
        Perform a feed forward of the neural network and get as fitness value th MSE(mean squared error)

        :param weights:  weights positions of a particle
        :param af: activation functions positions of a particle
        :param bias: bias positions of a particle
        :param _set: set on which to perform the feed forward
        :return: fitness value (mean squared error of nn)
        """
        return self.neural_network.feed_forward(weights, af, bias, _set)["mse"]

    def fit(self):
        # GLOBAL MSE PLOT VARIABLES INIT
        ###############################################
        k = 0
        x = []
        y = []
        ###############################################

        bar = Bar('Epoch ', max=self.epochs)
        for n in range(self.epochs):
            # Update particle's best
            for particle in self.particles:

                # Calculate fitness using particle's positions
                weights = particle.positions_weights.copy()
                af = particle.positions_af.copy()
                bias = particle.positions_bias.copy()
                particle_fitness = self.fitness(weights, af, bias, self.train_sets)

                # Calculate fitness using particle's best positions
                weights = particle.best_positions_weights.copy()
                af = particle.best_positions_af.copy()
                bias = particle.best_positions_bias.copy()
                particle_best_fitness = self.fitness(weights, af, bias, self.train_sets)

                # Update particle's best positions
                if particle_fitness < particle_best_fitness:
                    particle.fit = particle_fitness
                    particle.best_positions_weights = particle.positions_weights.copy()
                    particle.best_positions_af = particle.positions_af.copy()
                    particle.best_positions_bias = particle.positions_bias.copy()

            # Search global best particle
            for particle in self.particles:
                if particle.fit < self.global_best_fit or self.global_best_particle is None:
                    self.global_best_fit = particle.fit
                    self.global_best_particle = particle

            # Update each particle's informant
            for particle in self.particles:
                particle.set_informant(sample(self.particles, 4))  # TODO change to random number of sample

            # UPDATE GLOBAL BEST FITNESS PLOT VARIABLES
            ################################################
            k += 1
            x.append(k)
            y.append(self.global_best_fit)
            ################################################

            # Update each particle's velocity and positions
            for particle in self.particles:
                particle.update_velocities(self.alpha, self.beta, self.gamma, self.delta, self.global_best_particle)
                particle.update_positions()

            bar.next()
        bar.finish()
        plt.plot(x, y)
        plt.show()

    def predict(self):
        """
        Using the test set predict the outcome using the PSO global best particle positions
        The method is written ad hoc for sets with 1 and 2 inputs.
        Displays the graphs for expected and calculated outputs vs test set's inputs
        """
        # Get the outputs from neural network
        weights = self.global_best_particle.best_positions_weights.copy()
        af = self.global_best_particle.best_positions_af.copy()
        bias = self.global_best_particle.best_positions_bias.copy()
        ff = self.neural_network.feed_forward(weights, af, bias, self.test_sets)
        outs = ff["outputs"].ravel()

        # 2 input
        if self.test_sets[0].input.size > 1:
            in1 = []
            in2 = []
            exp = []
            for s in self.test_sets:
                in1.append(s.input[0])
                in2.append(s.input[1])
                exp.append(s.output)
            print("\tInput1\t|\tInput2\t|\tOutput\t|\tExpected\t")
            for n in range(len(outs)):
                print(f"\t{in1[n]}\t|\t{in2[n]}\t|\t{outs[n]:.4f}\t|\t{exp[n]}")

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(in1, in2, outs)
            ax.scatter(in1, in2, exp)
            ax.set_xlabel('x1')
            ax.set_ylabel('x2')
            ax.set_zlabel('y')
            plt.show()
        # 1 inputs
        else:
            ins = []
            exp = []
            for s in self.test_sets:
                ins.append(s.input)
                exp.append(s.output)
            print("\tInput\t|\tOutput\t|\tExpected\t")
            for n in range(len(outs)):
                print(f"\t{ins[n]}\t|\t{outs[n]:.4f}\t|\t{exp[n]}")
            plt.scatter(ins, outs)
            plt.scatter(ins, exp)
            plt.show()
        pass

# UNCOMMENT FOR TESTING
# if __name__ == "__main__":
# sets = NNSets("./Data/1in_linear.txt")
# nn = NeuralNetwork(sets.training_set)
# nn.create_layer(1, ActivationFunction.identity, 'input')
# nn.create_layer(4, ActivationFunction.sigmoid, 'hidden')
# nn.create_layer(1, ActivationFunction.step, 'output')
# pso = PSO(sets, 200, nn, 30, 1, 2, 1, 0)
# pso.fit()
# print("Best fitness val", pso.global_best_fit)
# print("Best Weights positions", pso.global_best_particle.best_positions_weights)
# print("Best Weights af", pso.global_best_particle.best_positions_weights)
# print("Best Weights bias", pso.global_best_particle.best_positions_weights)
# pso.predict()
