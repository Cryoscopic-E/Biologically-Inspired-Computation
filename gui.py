import tkinter as tk

from PIL import ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.image as mpimg

from activation_function import ActivationFunction
from draw_ann import DrawANN
from neural_network import NeuralNetwork
from nn_sets import NNSets
from particle_swarm import PSO

from matplotlib import style

style.use("ggplot")

root = tk.Tk()

# create a gui 1400*450
canvas1 = tk.Canvas(root, width=1400, height=450)
canvas1.pack()

# create a canvas image to place the background image
imagepath1 = r'./ann3.png'
image1 = ImageTk.PhotoImage(file=imagepath1)
canvas1.create_image(650, 200, image=image1)

# create a button to exit the application
button1 = tk.Button(root, text='Exit Application', command=root.destroy)
canvas1.create_window(85, 400, window=button1)


""" 
PSO: create all the widgets needed to modify the PSO
"""

# create box under description
canvas1.create_rectangle(35, 5, 480, 205, fill='lightblue')

# add title
canvas1.create_text(100, 15, text="Particle Swarm Optimisation", anchor="w", fill="blue")
canvas1.create_text(101, 15, text="Particle Swarm Optimisation", anchor="w", fill="red")

# add description of task
canvas1.create_text(45, 55, text="Change the parameters to see how they affects the performance of the neural network.",
                    anchor="w", width="410")

# PSO
# add explanation of parameters
canvas1.create_text(50, 90, text="Swarmsize", anchor="w", activefill="blue")
canvas1.create_text(50, 110, text="Alpha", anchor="w", activefill="blue")
canvas1.create_text(50, 130, text="Beta", anchor="w", activefill="blue")
canvas1.create_text(50, 150, text="Gamma", anchor="w", activefill="blue")
canvas1.create_text(50, 170, text="Delta", anchor="w", activefill="blue")

canvas1.create_text(140, 90, text="->")
canvas1.create_text(140, 110, text="->")
canvas1.create_text(140, 130, text="->")
canvas1.create_text(140, 150, text="->")
canvas1.create_text(140, 170, text="->")

canvas1.create_text(160, 90, text="desired swarm size", anchor="w", activefill="blue")
canvas1.create_text(160, 110, text="propotion of velocity to be retained", anchor="w", activefill="blue")
canvas1.create_text(160, 130, text="proportion of personal best to be retained", anchor="w", activefill="blue")
canvas1.create_text(160, 150, text="proportion of the informants' best to be retained", anchor="w", activefill="blue")
canvas1.create_text(160, 170, text="proportion of global best to be retained", anchor="w", activefill="blue")

# add entry for parameters
entry1 = tk.Entry(root)
canvas1.create_window(230, 220, window=entry1)
canvas1.create_text(50, 220, text="Swarmsize", anchor="w")

entry2 = tk.Entry(root)
canvas1.create_window(230, 240, window=entry2)
canvas1.create_text(50, 240, text="Alpha", anchor="w")

entry3 = tk.Entry(root)
canvas1.create_window(230, 260, window=entry3)
canvas1.create_text(50, 260, text="Beta", anchor="w")

entry4 = tk.Entry(root)
canvas1.create_window(230, 280, window=entry4)
canvas1.create_text(50, 280, text="Gamma", anchor="w")

entry5 = tk.Entry(root)
canvas1.create_window(230, 300, window=entry5)
canvas1.create_text(50, 300, text="Delta", anchor="w")

entry6 = tk.Entry(root)
canvas1.create_window(230, 320, window=entry6)
canvas1.create_text(50, 320, text="Epochs", anchor="w")

""" 
ANN: create all the widgets needed to modify the ANN
"""

# create box under description
canvas1.create_rectangle(530, 5, 940, 110, fill='lightblue')

# add title
canvas1.create_text(630, 15, text="ANN Architecture", anchor="w", fill="blue")
canvas1.create_text(631, 15, text="ANN Architecture", anchor="w", fill="red")

# add description
canvas1.create_text(540, 55,
                    text="Task: add hidden layer, decide how many neurons per layer and select the activation function",
                    anchor="w", width="410")

# create entres for the Ann's parameters
layer5 = tk.Entry(root)
layer4 = tk.Entry(root)
layer3 = tk.Entry(root)
layer2 = tk.Entry(root)
layer1 = tk.Entry(root)

act_func1 = tk.StringVar(root)
act_func2 = tk.StringVar(root)
act_func3 = tk.StringVar(root)
act_func4 = tk.StringVar(root)
act_func5 = tk.StringVar(root)
data_selected = tk.StringVar(root)

dataset_selected = tk.StringVar(root)


def add_box_5():
    """
    Add a box (layer) to the existing ANN architecture
    :return:
    """
    canvas1.create_window(650, 250, window=layer5, anchor="w")
    activation_fun = ['-', 'Null', 'Sigmoid', 'Hyperbolic tangent', 'Cosine', 'Gaussian']
    act_func5.set(activation_fun[0])  # default value
    act_func_1 = tk.OptionMenu(canvas1, act_func5, *activation_fun)
    canvas1.create_window(820, 250, window=act_func_1, anchor="w")


def add_box_4():
    """
    Add a box (layer) to the existing ANN architecture
    :return:
    """
    canvas1.create_window(650, 220, window=layer4, anchor="w")
    button6 = tk.Button(root, text='Add Layer', fg="Red", command=add_box_5)
    canvas1.create_window(550, 220, window=button6, anchor="w")
    activation_fun = [
        '-', 'Null', 'Sigmoid', 'Hyperbolic tangent', 'Cosine', 'Gaussian'
    ]
    act_func4.set(activation_fun[0])  # default value
    act_func_1 = tk.OptionMenu(canvas1, act_func4, *activation_fun)
    canvas1.create_window(820, 220, window=act_func_1, anchor="w")


def add_box_3():
    """
    Add a box (layer) to the existing ANN architecture
    :return:
    """
    canvas1.create_window(650, 190, window=layer3, anchor="w")
    button6 = tk.Button(root, text='Add Layer', fg="Red", command=add_box_4)
    canvas1.create_window(550, 190, window=button6, anchor="w")
    activation_fun = ['-', 'Null', 'Sigmoid', 'Hyperbolic tangent', 'Cosine', 'Gaussian']
    act_func3.set(activation_fun[0])  # default value
    act_func_1 = tk.OptionMenu(canvas1, act_func3, *activation_fun)
    canvas1.create_window(820, 190, window=act_func_1, anchor="w")


def add_box_2():
    """
    Add a box (layer) to the existing ANN architecture
    :return:
    """
    canvas1.create_window(650, 160, window=layer2, anchor="w")
    button5 = tk.Button(root, text='Add Layer', fg="Red",
                        command=add_box_3)
    canvas1.create_window(550, 160, window=button5, anchor="w")
    activation_fun = ['-', 'Null', 'Sigmoid', 'Hyperbolic tangent', 'Cosine', 'Gaussian']
    act_func2.set(activation_fun[0])  # default value
    act_func_1 = tk.OptionMenu(canvas1, act_func2, *activation_fun)
    canvas1.create_window(820, 160, window=act_func_1, anchor="w")


def add_box():
    """
    Add a box (layer) to the existing ANN architecture.
    Add the possibility to select the dataset to use
    :return:
    """
    canvas1.create_window(650, 130, window=layer1, anchor="w")
    button4 = tk.Button(root, text='Add Layer', fg="Red", command=add_box_2)
    canvas1.create_window(550, 130, window=button4, anchor="w")
    activation_fun = ['-', 'Null', 'Sigmoid', 'Hyperbolic tangent', 'Cosine', 'Gaussian']

    act_func1.set(activation_fun[0])  # default value
    act_func_1 = tk.OptionMenu(canvas1, act_func1, *activation_fun)
    canvas1.create_window(820, 130, window=act_func_1, anchor="w")

    # Type of dataset to use
    canvas1.create_rectangle(1000, 7, 1300, 45, fill='lightblue')

    canvas1.create_text(1010, 25, text="Which function to approximate? ", anchor="w", fill="blue")
    canvas1.create_text(1011, 25, text="Which function to approximate? ", anchor="w", fill="red")

    dataset = ['-', 'Linear', 'Cubic', 'Sine', 'Tanh', 'Complex', 'Xor']

    data_selected.set(dataset[0])  # default value
    data_selected_1 = tk.OptionMenu(canvas1, data_selected, *dataset)
    canvas1.create_window(1010, 70, window=data_selected_1, anchor="w")

# button to start building the ANN architecture
button3 = tk.Button(root, text='Start!', fg="Red", command=add_box, anchor="w")
canvas1.create_window(700, 90, window=button3)


def insert_graph_and_run_application():
    """
    Function called when "Click to Train the ANN!" button is pressed.
    Run the application and displays the graphs created
    :return:
    """

    global x1
    global x2
    global x3
    global x4
    global x5

    # create a vector of input to draw and create the architecture of the ANN
    draw_input = []

    try:
        x1 = int(layer1.get())
        draw_input.append(x1)
    except ValueError:
        x1 = 0

    try:
        x2 = int(layer2.get())
        draw_input.append(x2)
    except ValueError:
        x2 = 0

    try:
        x3 = int(layer3.get())
        draw_input.append(x3)
    except ValueError:
        x3 = 0

    try:
        x4 = int(layer4.get())
        draw_input.append(x4)
    except ValueError:
        x4 = 0

    try:
        x5 = int(layer5.get())
        draw_input.append(x5)
    except ValueError:
        x5 = 0

    var = draw_input
    var.append(1)
    x_input = [x1, x2, x3, x4, x5]


    # load all the datasets in 6 variables
    sets_linear = NNSets("./Data/1in_linear.txt")
    sets_cubic = NNSets("./Data/1in_cubic.txt")
    sets_sine = NNSets("./Data/1in_sine.txt")
    sets_tanh = NNSets("./Data/1in_tanh.txt")
    sets_complex = NNSets("./Data/2in_complex.txt")
    sets_xor = NNSets("./Data/2in_xor.txt")

    # assign the dataset to the option selected in the gui for the dataset
    if data_selected.get() == "Linear":
        dat_sel = sets_linear
        n_neurons = 1

    if data_selected.get() == "Cubic":
        dat_sel = sets_cubic
        n_neurons = 1

    if data_selected.get() == "Sine":
        dat_sel = sets_sine
        n_neurons = 1

    if data_selected.get() == "Tanh":
        dat_sel = sets_tanh
        n_neurons = 1

    if data_selected.get() == "Complex":
        dat_sel = sets_complex
        n_neurons = 2

    if data_selected.get() == "Xor":
        dat_sel = sets_xor
        n_neurons = 2

    # get the information from the entries to feed the ANN and PSO
    act_entries = [act_func1.get(), act_func2.get(), act_func3.get(), act_func4.get(), act_func5.get()]
    pso_entries = [int(entry1.get()), float(entry2.get()), float(entry3.get()), float(entry4.get()),
                   float(entry5.get()),
                   int(entry6.get())]
    actfun = []

    for af in act_entries:
        if af == "":
            af = 0
        if af == "Null":
            af = ActivationFunction.null
            actfun.append(af)
        if af == "Sigmoid":
            af = ActivationFunction.sigmoid
            actfun.append(af)
        if af == "Hyperbolic tangent":
            af = ActivationFunction.hyperbolic_tangent
            actfun.append(af)
        if af == "Cosine":
            af = ActivationFunction.cosine
            actfun.append(af)
        if af == "Gaussian":
            af = ActivationFunction.gaussian
            actfun.append(af)

    # ----------- START ANN -------------
    nn = NeuralNetwork(dat_sel.training_set)
    nn.create_layer(n_neurons, ActivationFunction.identity, 'input')

    for x, af in zip(x_input, actfun):
        nn.create_layer(x, af, 'hidden')

    nn.create_layer(1, ActivationFunction.identity, 'output')

    pso = PSO(dat_sel, pso_entries[5], nn, pso_entries[0], pso_entries[1],
              pso_entries[2], pso_entries[3], pso_entries[4])
    pso.fit()
    print("Best fitness val", pso.global_best_fit)
    print("Best Weights positions", pso.global_best_particle.best_positions_weights)
    print("Best Weights af", pso.global_best_particle.best_positions_weights)
    print("Best Weights bias", pso.global_best_particle.best_positions_weights)
    pso.predict()

    # --------------- GRAPHS -----------------

    # GRAPH ONE
    draw = []

    draw.append(n_neurons)
    for i in var:
        draw.append(i)
    var = draw[::-1]
    ann = DrawANN(var)
    ann.draw_ann()
    img_arr = mpimg.imread('ann.jpg')
    figure1 = Figure(figsize=(14, 13), dpi=33)
    subplot = figure1.add_subplot(111)
    subplot.axis('off')
    subplot.imshow(img_arr)
    canvas1 = FigureCanvasTkAgg(figure1, master=root)
    canvas1._tkcanvas.pack(side="left", fill="both", expand=0)

    # GRAPH TWO
    img_arr1 = mpimg.imread('predict.jpg')
    figure2 = Figure(figsize=(14, 13), dpi=33)
    subplot = figure2.add_subplot(111)
    subplot.axis('off')
    subplot.imshow(img_arr1)
    canvas1 = FigureCanvasTkAgg(figure2, master=root)
    art = canvas1._tkcanvas.pack(side="right", fill="both", expand=0)

    # GRAPH THREE
    img_arr2 = mpimg.imread('mse_error.jpg')
    figure3 = Figure(figsize=(14, 13), dpi=34)
    subplot = figure3.add_subplot(111)
    subplot.axis('off')
    subplot.imshow(img_arr2)
    canvas1 = FigureCanvasTkAgg(figure3, master=root)
    canvas1._tkcanvas.pack(side="right", fill="both", expand=0)


button2 = tk.Button(root, text='Click to Train the ANN!', command=insert_graph_and_run_application, anchor="w")
canvas1.create_window(110, 370, window=button2)

root.mainloop()