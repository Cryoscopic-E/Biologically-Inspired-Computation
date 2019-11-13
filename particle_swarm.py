from random import randrange, choice
from neural_network import NeuralNetwork


class Particle:
    def __init__(self, dimensions):
        self._positions = [randrange(-2.0, 2.0) for n in range(dimensions)]
        self._velocity = [randrange(-1.0, 1.0) for n in range(dimensions)]
        self._best_pos = self._positions
        self.informant = None
        pass

    def update_positions(self):
        """
        Update particle's position using the velocity vector
        """
        for n in range(len(self._positions)):
            self._positions[n] += self._velocity[n]
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
                              (beta * (self._best_pos[dim] - self._positions[dim])) + \
                              (gamma * (self.informant.bestPositions[dim] - self._positions[dim])) + \
                              (delta * (global_best[dim] - self._positions[dim]))
        pass

    def get_positions(self):
        return self._positions

    def set_best_pos(self):
        self._best_pos = self._positions
        pass

    def set_informant(self, particle):
        self.informant = particle
        pass


class PSO:
    def __init__(self, swarm_size, dimensions, alpha, beta, gamma, delta):
        self.global_best_positions = None
        self.dimensions = dimensions
        self.particles = [Particle(dimensions) for n in range(swarm_size)]
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

    def fitness(self, positions):
        """
        Perform a feed forward of the neural network using the n particle positions as weights
        :param positions: position list of particle n
        :return: fitness value (squared mean error of nn)
        """
        return 2

    def update(self):
        for n in range(100):
            for particle in self.particles:
                particle_fitness = self.fitness(particle.get_positions())
                if self.global_best_positions is None:
                    self.global_best_positions = particle.get_positions()

                global_fitness = self.fitness(self.global_best_positions)

                if particle_fitness < global_fitness:
                    particle.set_best_pos()
                    self.global_best_positions = particle.get_positions()

                for d in range(self.dimensions):
                    b = randrange(0, self.beta)
                    c = randrange(0, self.gamma)
                    d = randrange(0, self.delta)
                    particle.update_velocity(d, self.alpha, b, c, d, None, self.global_best_positions)

                particle.update_positions()
