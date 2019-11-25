"""
Saves the experiments results in txt files.
The name of the txt files is given by the combination of parameters used to
create and train the ANN and PSO

"""

import numpy as np

from activation_function import ActivationFunction
from neural_network import NeuralNetwork
from nn_sets import NNSets
from particle_swarm import PSO



def main():

    # load data sets in different varaibles
    sets_linear = NNSets("./Data/1in_linear.txt")
    sets_cubic = NNSets("./Data/1in_cubic.txt")
    sets_sine = NNSets("./Data/1in_sine.txt")
    sets_tanh = NNSets("./Data/1in_tanh.txt")
    sets_complex = NNSets("./Data/2in_complex.txt")
    sets_xor = NNSets("./Data/2in_xor.txt")

    # set parameters to test
    n_neurons = [1, 3, 5]
    swarmsize = [10, 25, 40]
    epochs = [50, 150, 300]
    sets_1in = [sets_linear, sets_cubic, sets_sine, sets_tanh]
    sets_1in_txt = ["sets_linear", "sets_cubic", "sets_sine", "sets_tanh"]
    sets_2in = [sets_xor, sets_complex]
    sets_2in_txt = ["sets_xor", "sets_complex"]
    hidden_layers = [1, 2]

    # create the lists to store values for ten experiments
    best_val = []
    best_weights_position = []
    best_weights_af = []
    best_weights_bias = []

    # start loop for experiments with 1 input
    for set, set_txt in zip(sets_1in, sets_1in_txt):
        for neurons in n_neurons:
            for layers in hidden_layers:
                for size in swarmsize:
                    for epoch in epochs:
                        for j in range(1, 11):

                            # create ANN architecture
                            sets = set
                            nn = NeuralNetwork(sets.training_set)
                            nn.create_layer(1, ActivationFunction.identity, 'input')
                            i = 0
                            while i > layers:
                                nn.create_layer(neurons, ActivationFunction.sigmoid, 'hidden')
                                i += 1
                            nn.create_layer(1, ActivationFunction.step, 'output')

                            # create PSO
                            pso = PSO(sets, epoch, nn, size, 1, 2, 1, 0)
                            pso.fit()

                            # use append to create the lists of values needed
                            best_val.append(pso.global_best_fit)
                            best_weights_position.append(pso.global_best_particle.best_positions_weights)
                            best_weights_af.append(pso.global_best_particle.best_positions_af)
                            best_weights_bias.append(pso.global_best_particle.best_positions_bias)


                        # find mean of bests values
                        mean_best_val = np.mean(best_val, axis=0)
                        mean_best_weights_position = np.mean(best_weights_position, axis=0)
                        mean_best_weights_af = np.mean(best_weights_af, axis=0)
                        mean_best_weights_bias = np.mean(best_weights_bias, axis=0)


                        # return results of pso.predict() to save in txt
                        outs = pso.predict()

                        ins = []
                        exp = []
                        for s in pso.test_sets:
                            ins.append(s.input)
                            exp.append(s.output)

                        # save outputs in txt files
                        file = open(
                            "file_" + set_txt + "_neurons:_" + str(neurons) + "_hidden_layers:_"+ str(layers) + "_swarsize:_" + str(size) + "_epochs:_" + str(
                                epoch) + ".txt", "w")
                        file.writelines(["Best fitness val: " + str(mean_best_val),
                                         "\nBest Weights positions: " + str(mean_best_weights_position),
                                         "\nBest Weights af: " + str(mean_best_weights_af),
                                         "\nBest Weights bias: " + str(mean_best_weights_bias),
                                         "\n",
                                         "\tInput\t|\tOutput\t|\tExpected\t",
                                         "\n",
                                         "\t"+ str(ins[0]) +"\t|\t" + str(outs[0]) + "\t|\t" + str(exp[0]),
                                         "\n", "\t"+ str(ins[1]) +"\t|\t" + str(outs[1]) + "\t|\t" + str(exp[1]),
                                         "\n", "\t"+ str(ins[2]) +"\t|\t" + str(outs[2]) + "\t|\t" + str(exp[2]),
                                         "\n", "\t"+ str(ins[3]) +"\t|\t" + str(outs[3]) + "\t|\t" + str(exp[3]),
                                         "\n", "\t"+ str(ins[4]) +"\t|\t" + str(outs[4]) + "\t|\t" + str(exp[4]),
                                         "\n", "\t"+ str(ins[5]) +"\t|\t" + str(outs[5]) + "\t|\t" + str(exp[5]),
                                         "\n", "\t"+ str(ins[6]) +"\t|\t" + str(outs[6]) + "\t|\t" + str(exp[6]),
                                         "\n", "\t"+ str(ins[7]) +"\t|\t" + str(outs[7]) + "\t|\t" + str(exp[7]),
                                         "\n", "\t"+ str(ins[8]) +"\t|\t" + str(outs[8]) + "\t|\t" + str(exp[8])])
                        file.close()

                        best_val = []
                        best_weights_position = []
                        best_weights_af = []
                        best_weights_bias = []


    # start loop for experiments with 2 inputs
    for set, txt_set in zip(sets_2in, sets_2in_txt):
        for neurons in n_neurons:
            for layers in hidden_layers:
                for size in swarmsize:
                    for epoch in epochs:
                        for j in range(1, 11):
                            sets = set
                            nn = NeuralNetwork(sets.training_set)
                            nn.create_layer(2, ActivationFunction.identity, 'input')
                            i = 0
                            while i > layers:
                                nn.create_layer(neurons, ActivationFunction.sigmoid, 'hidden')
                                i += 1
                            nn.create_layer(1, ActivationFunction.step, 'output')
                            pso = PSO(sets, epoch, nn, size, 1, 2, 1, 0)
                            pso.fit()
                            outs = pso.predict()

                            # use append to create the lists of values needed
                            best_val.append(pso.global_best_fit)
                            best_weights_position.append(pso.global_best_particle.best_positions_weights)
                            best_weights_af.append(pso.global_best_particle.best_positions_af)
                            best_weights_bias.append(pso.global_best_particle.best_positions_bias)

                        # find mean of bests values
                        mean_best_val = np.mean(best_val, axis=0)
                        mean_best_weights_position = np.mean(best_weights_position, axis=0)
                        mean_best_weights_af = np.mean(best_weights_af, axis=0)
                        mean_best_weights_bias = np.mean(best_weights_bias, axis=0)

                        in1 = []
                        in2 = []
                        exp = []
                        for s in pso.test_sets:
                            in1.append(s.input[0])
                            in2.append(s.input[1])
                            exp.append(s.output)

                        file = open("file_" + txt_set + "_neurons:_" + str(neurons) + "_hidden_layers:_"+ str(layers) + "_swarsize:_"  + str(size) + "_epochs:_"  + str(epoch) + ".txt", "w")
                        file.writelines(["Best fitness val: " + str(mean_best_val),
                                         "\nBest Weights positions: " + str(mean_best_weights_position),
                                         "\nBest Weights af: " + str(mean_best_weights_af),
                                         "\nBest Weights bias: " + str(mean_best_weights_bias),
                                         "\n",
                                         "\tInput1\t|\tInput2\t|\tOutput\t|\tExpected\t",
                                         "\n", "\t" + str(in1[0]) + "\t|\t" + str(in2[0]) + "\t|\t\t" + str(outs[0]) + "\t|\t" + str(exp[0]),
                                         "\n", "\t" + str(in1[1]) + "\t|\t" + str(in2[1]) + "\t|\t" + str(outs[1]) + "\t|\t" + str(exp[1]),
                                         "\n", "\t" + str(in1[2]) + "\t|\t" + str(in2[2]) + "\t|\t" + str(outs[2]) + "\t|\t" + str(exp[2]),
                                         "\n", "\t" + str(in1[3]) + "\t|\t" + str(in2[3]) + "\t|\t" + str(outs[3]) + "\t|\t" + str(exp[3]),
                                         "\n", "\t" + str(in1[4]) + "\t|\t" + str(in2[4]) + "\t|\t" + str(outs[4]) + "\t|\t" + str(exp[4]),
                                         "\n", "\t" + str(in1[5]) + "\t|\t" + str(in2[5]) + "\t|\t" + str(outs[5]) + "\t|\t" + str(exp[5]),
                                         "\n", "\t" + str(in1[6]) + "\t|\t" + str(in2[6]) + "\t|\t" + str(outs[6]) + "\t|\t" + str(exp[6]),
                                         "\n", "\t" + str(in1[7]) + "\t|\t" + str(in2[7]) + "\t|\t" + str(outs[7]) + "\t|\t" + str(exp[7]),
                                         "\n", "\t" + str(in1[8]) + "\t|\t" + str(in2[8]) + "\t|\t" + str(outs[8]) + "\t|\t" + str(exp[8])])
                        file.close()

                        best_val = []
                        best_weights_position = []
                        best_weights_af = []
                        best_weights_bias = []


if __name__ == "__main__":
    main()

