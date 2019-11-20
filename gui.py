import tkinter as tk

from PIL import ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

root = tk.Tk()

canvas1 = tk.Canvas(root, width=1000, height=450)
canvas1.pack()

imagepath1 = r'./ann1'  # include the path for the image (use 'r' before the path string to address any special character such as '\'. Also, make sure to include the image type - here it is jpg)
image1 = ImageTk.PhotoImage(file=imagepath1)  # PIL module
canvas1.create_image(610, 250, image=image1)  # create a canvas image to place the background image

button1 = tk.Button(root, text='Exit Application', command=root.destroy)
canvas1.create_window(85, 400, window=button1)

""" PSO """

# create box under description
# canvas1.create_rectangle(5, 5, 455, 345, fill='', activefill="lightgreen")
canvas1.create_rectangle(7, 5, 450, 205, fill='lightblue')

# add title
canvas1.create_text(100, 15, text="Particle Swarm Optimisation", anchor="w", fill="blue")
canvas1.create_text(101, 15, text="Particle Swarm Optimisation", anchor="w", fill="red")

canvas1.create_text(15, 55, text="Change the parameters to see how they affects the performance of the neural network.", anchor="w", width="410")

# PSO
#add explanation
canvas1.create_text(20, 90, text="Swarmsize", anchor="w", activefill="blue")
canvas1.create_text(20, 110, text="Alpha", anchor="w", activefill="blue")
canvas1.create_text(20, 130, text="Beta", anchor="w", activefill="blue")
canvas1.create_text(20, 150, text="Gamma", anchor="w", activefill="blue")
canvas1.create_text(20, 170, text="Delta", anchor="w", activefill="blue")
canvas1.create_text(20, 190, text="Epsilon", anchor="w", activefill="blue")

canvas1.create_text(110, 90, text="->")
canvas1.create_text(110, 110, text="->")
canvas1.create_text(110, 130, text="->")
canvas1.create_text(110, 150, text="->")
canvas1.create_text(110, 170, text="->")
canvas1.create_text(110, 190, text="->")

canvas1.create_text(130, 90, text="desired swarm size", anchor="w", activefill="blue")
canvas1.create_text(130, 110, text="propotion of velocity to be retained", anchor="w", activefill="blue")
canvas1.create_text(130, 130, text="proportion of personal best to be retained", anchor="w", activefill="blue")
canvas1.create_text(130, 150, text="proportion of the informants' best to be retained", anchor="w", activefill="blue")
canvas1.create_text(130, 170, text="proportion of global best to be retained", anchor="w", activefill="blue")
canvas1.create_text(130, 190, text="jump size of a particle", anchor="w", activefill="blue")

# ANN
#add explanation
# canvas1.create_text(20, 50, text="Swarmsize", anchor="w", activefill="blue")
# canvas1.create_text(20, 70, text="Alpha", anchor="w", activefill="blue")
# canvas1.create_text(20, 90, text="Beta", anchor="w", activefill="blue")
# canvas1.create_text(20, 110, text="Gamma", anchor="w", activefill="blue")
# canvas1.create_text(20, 130, text="Delta", anchor="w", activefill="blue")
# canvas1.create_text(20, 150, text="Epsilon", anchor="w", activefill="blue")


entry1 = tk.Entry(root)
canvas1.create_window(200, 220, window=entry1)
canvas1.create_text(20, 220, text="Swarmsize", anchor="w")

entry2 = tk.Entry(root)
canvas1.create_window(200, 240, window=entry2)
canvas1.create_text(20, 240, text="Alpha", anchor="w")

entry3 = tk.Entry(root)
canvas1.create_window(200, 260, window=entry3)
canvas1.create_text(20, 260, text="Beta", anchor="w")

entry4 = tk.Entry(root)
canvas1.create_window(200, 280, window=entry4)
canvas1.create_text(20, 280, text="Gamma", anchor="w")

entry5 = tk.Entry(root)
canvas1.create_window(200, 300, window=entry5)
canvas1.create_text(20, 300, text="Delta", anchor="w")

entry6 = tk.Entry(root)
canvas1.create_window(200, 320, window=entry6)
canvas1.create_text(20, 320, text="Epsilon", anchor="w")


""" ANN """

# create box under description
canvas1.create_rectangle(500, 5, 900, 110, fill='lightblue')

# add title
canvas1.create_text(600, 15, text="ANN Architecture", anchor="w", fill="blue")
canvas1.create_text(601, 15, text="ANN Architecture", anchor="w", fill="red")

canvas1.create_text(510, 55, text="Task: add hidden layer, decide how many neurons per layer and select the activation function", anchor="w", width="410")


i = 160

def add_box_5():
    ent03 = tk.Entry(root)
    canvas1.create_window(620, 250, window=ent03, anchor="w")
    OPTIONS = [
        '-', 'Null', 'Hyperbolic tangent', 'Cosine', 'Gaussian'
    ]
    act_func = tk.StringVar()
    act_func.set(OPTIONS[0])  # default value
    act_func_1 = tk.OptionMenu(canvas1, act_func, *OPTIONS)
    canvas1.create_window(790, 250, window=act_func_1, anchor="w")

def add_box_4():
    ent02 = tk.Entry(root)
    canvas1.create_window(620, 220, window=ent02, anchor="w")
    button6 = tk.Button(root, text='Add Layer', fg="Red", command=add_box_5)
    canvas1.create_window(520, 220, window=button6, anchor="w")
    OPTIONS = [
        '-', 'Null', 'Hyperbolic tangent', 'Cosine', 'Gaussian'
    ]
    act_func = tk.StringVar()
    act_func.set(OPTIONS[0])  # default value
    act_func_1 = tk.OptionMenu(canvas1, act_func, *OPTIONS)
    canvas1.create_window(790, 220, window=act_func_1, anchor="w")


def add_box_3():
    ent01 = tk.Entry(root)
    canvas1.create_window(620, 190, window=ent01, anchor="w")
    button6 = tk.Button(root, text='Add Layer', fg="Red", command=add_box_4)
    canvas1.create_window(520, 190, window=button6, anchor="w")
    OPTIONS = [
        '-', 'Null', 'Hyperbolic tangent', 'Cosine', 'Gaussian'
    ]
    act_func = tk.StringVar()
    act_func.set(OPTIONS[0])  # default value
    act_func_1 = tk.OptionMenu(canvas1, act_func, *OPTIONS)
    canvas1.create_window(790, 190, window=act_func_1, anchor="w")


def add_box_2():
    ent1 = tk.Entry(root)
    canvas1.create_window(620, 160, window=ent1, anchor="w")
    button5 = tk.Button(root, text='Add Layer', fg="Red", command=add_box_3)
    canvas1.create_window(520, 160, window=button5, anchor="w")
    OPTIONS = [
        '-', 'Null', 'Hyperbolic tangent', 'Cosine', 'Gaussian'
    ]
    act_func = tk.StringVar()
    act_func.set(OPTIONS[0])  # default value
    act_func_1 = tk.OptionMenu(canvas1, act_func, *OPTIONS)
    canvas1.create_window(790, 160, window=act_func_1, anchor="w")

def add_box():
    ent = tk.Entry(root)
    canvas1.create_window(620, 130, window=ent, anchor="w")
    button4 = tk.Button(root, text='Add Layer', fg="Red", command=add_box_2)
    canvas1.create_window(520, 130, window=button4, anchor="w")
    OPTIONS = [
        '-', 'Null', 'Hyperbolic tangent', 'Cosine', 'Gaussian'
    ]
    act_func = tk.StringVar()
    act_func.set(OPTIONS[0])  # default value
    act_func_1 = tk.OptionMenu(canvas1, act_func, *OPTIONS)
    canvas1.create_window(790, 130, window=act_func_1, anchor="w")


button3 = tk.Button(root, text='Start!', fg="Red", command=add_box, anchor="w")
canvas1.create_window(700, 90, window=button3)


def insert_number():
    global x1
    global x2
    global x3
    x1 = float(entry1.get())
    x2 = float(entry2.get())
    x3 = float(entry3.get())

    figure1 = Figure(figsize=(5, 4), dpi=100)
    subplot1 = figure1.add_subplot(111)
    xAxis = [float(x1), float(x2), float(x3)]
    yAxis = [float(x1), float(x2), float(x3)]
    subplot1.bar(xAxis, yAxis, color='g')
    bar1 = FigureCanvasTkAgg(figure1, root)
    bar1.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=0)

    figure2 = Figure(figsize=(5, 4), dpi=100)
    subplot2 = figure2.add_subplot(111)
    labels2 = 'Label1', 'Label2', 'Label3'
    pieSizes = [float(x1), float(x2), float(x3)]
    explode2 = (0, 0.1, 0)
    subplot2.pie(pieSizes, explode=explode2, labels=labels2, autopct='%1.1f%%', shadow=True, startangle=90)
    subplot2.axis('equal')
    pie2 = FigureCanvasTkAgg(figure2, root)
    pie2.get_tk_widget().pack()




button2 = tk.Button(root, text='Click to Train the ANN! ', command=insert_number, anchor="w")
canvas1.create_window(110, 370, window=button2)

root.mainloop()