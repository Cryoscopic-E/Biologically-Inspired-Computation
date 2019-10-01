import os

def get_InOut_from_file(txt_file):
    f = open(txt_file, "r")
    lines = f.readlines()

    input = []
    output = []

    for line in lines:

        data = line.split()
        if len(data) == 3:
            input.append([data[0], data[1]])
            output.append(data[2])
        if len(data) == 2:
            input.append(data[0])
            output.append(data[1])

    #print(input)
    #print(output)
    return input, output




def main():
    root_path = os.path.dirname(os.path.abspath(__file__))

    #store paths
    f_cubic = os.path.join(root_path, "Data", "1in_cubic.txt")
    # f_linear = os.path.join(root_path, "Data", "1in_linear.txt")
    # f_sine = os.path.join(root_path, "Data", "1in_sine.txt")
    # f_tanh = os.path.join(root_path, "Data", "1in_tanh.txt")
    # f_complex = os.path.join(root_path, "Data", "2in_complex.txt")
    # f_xor = os.path.join(root_path, "Data", "2in_xor.txt")

    #print(input_path)
    input_cubic, output_cubic = get_InOut_from_file(f_cubic)
    # input_linear, output_linear = get_InOut_from_file(f_linear)
    # input_sine, output_sine = get_InOut_from_file(f_sine)
    # input_tanh, output_tanh =  get_InOut_from_file(f_tanh)
    # input_complex, output_complex = get_InOut_from_file(f_complex)
    # input_xor, output_xor = get_InOut_from_file(f_xor)

    #print(input_cubic)
    #print(output_cubic)

if __name__ == "__main__":
    main()