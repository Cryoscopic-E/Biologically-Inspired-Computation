"""
Salva i riusltati degli esperimenti su
dei fogli txt il cui nome è dato dalla combianzione di epoche,numero di neuroni,
dataset utilizzato e swarmsize.

DA FARE:
- aggiungere linea per MSE e qualche coppia di in/out come accade nel terminale
- aggiungere hidden layer alla fine dei loop (----ora cerco di automatizzare questo---)

"""


from activation_function import ActivationFunction
from neural_network import NeuralNetwork
from nn_sets import NNSets
from particle_swarm import PSO


def main():

    sets_linear = NNSets("./Data/1in_linear.txt")
    sets_cubic = NNSets("./Data/1in_cubic.txt")
    sets_sine = NNSets("./Data/1in_sine.txt")
    sets_tanh = NNSets("./Data/1in_tanh.txt")

    sets_complex = NNSets("./Data/2in_complex.txt")
    sets_xor = NNSets("./Data/2in_xor.txt")

    n_neurons = [1, 2, 3, 5, 10, 20]
    swarmsize = [5, 10, 15, 20, 25, 30, 40, 50]
    epochs = [1, 2, 3, 4, 5, 10, 20, 30, 50, 75, 100, 150, 200]
    sets_1in = [sets_linear, sets_cubic, sets_sine, sets_tanh]
    sets_1in_txt = ["sets_linear", "sets_cubic", "sets_sine", "sets_tanh"]
    sets_2in = [sets_complex, sets_xor]
    sets_2in_txt = ["sets_complex", "sets_xor"]


    for set, set_txt in zip(sets_1in, sets_1in_txt):
        for neurons in n_neurons:
            for size in swarmsize:
                for epoch in epochs:
                    sets = set
                    nn = NeuralNetwork(sets.training_set)
                    nn.create_layer(1, ActivationFunction.identity, 'input')
                    nn.create_layer(neurons, ActivationFunction.sigmoid, 'hidden')
                    nn.create_layer(1, ActivationFunction.step, 'output')
                    pso = PSO(sets, epoch, nn, size, 1, 2, 1, 0)
                    pso.fit()
                    pso.predict()
                    file = open(
                        "file_" + set_txt + "_neurons:_" + str(neurons) + "_swarsize:_" + str(size) + "_epochs:_" + str(
                            epoch) + ".txt", "w")
                    file.writelines(["Best fitness val: " + str(pso.global_best_fit),
                                     "\nBest Weights positions: " + str(
                                         pso.global_best_particle.best_positions_weights),
                                     "\nBest Weights af: " + str(pso.global_best_particle.best_positions_weights),
                                     "\nBest Weights bias: " + str(pso.global_best_particle.best_positions_weights),
                                     "\nMSE error: " + str(nn.mse)])
                    file.close()
                    print("done")


    for set, txt_set in zip(sets_2in, sets_2in_txt):
        for neurons in n_neurons:
            for size in swarmsize:
                for epoch in epochs:
                    sets = set
                    nn = NeuralNetwork(sets.training_set)
                    nn.create_layer(2, ActivationFunction.identity, 'input')
                    nn.create_layer(neurons, ActivationFunction.sigmoid, 'hidden')
                    nn.create_layer(1, ActivationFunction.step, 'output')
                    pso = PSO(sets, epoch, nn, size, 1, 2, 1, 0)
                    fit = pso.fit()
                    prediction = pso.predict()
                    file = open("file_" + txt_set+ "_neurons:_" + str(neurons) + "_swarsize:_"  + str(size) + "_epochs:_"  + str(epoch) + ".txt", "w")
                    file.writelines(["Best fitness val: " + str(pso.global_best_fit),
                                     "\nBest Weights positions: " + str(pso.global_best_particle.best_positions_weights),
                                     "\nBest Weights af: " + str(pso.global_best_particle.best_positions_weights),
                                     "\nBest Weights bias: " + str(pso.global_best_particle.best_positions_weights),
                                     "\nMSE error: " + str(nn.mse)])
                    file.close()
                    print("done")



if __name__ == "__main__":
    main()

