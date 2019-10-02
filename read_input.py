# import os

def get_InOut_from_file(txt_file):
    f = open(txt_file, "r")
    lines = f.readlines()
    data = []
    for line in lines:

        row_split = line.split()
        data.append(row_split)
    #print(data)
    return data


#
# def main():
#     root_path = os.path.dirname(os.path.abspath(__file__))
#
#     #store paths
#     f_cubic = os.path.join(root_path, "Data", "1in_cubic.txt")
#     # f_linear = os.path.join(root_path, "Data", "1in_linear.txt")
#     # f_sine = os.path.join(root_path, "Data", "1in_sine.txt")
#     # f_tanh = os.path.join(root_path, "Data", "1in_tanh.txt")
#     #f_complex = os.path.join(root_path, "Data", "2in_complex.txt")
#     # f_xor = os.path.join(root_path, "Data", "2in_xor.txt")
#
#     #print(input_path)
#     data_cubic = get_InOut_from_file(f_cubic)
#     # data_linear = get_InOut_from_file(f_linear)
#     # data_sine = get_InOut_from_file(f_sine)
#     # data_tanh =  get_InOut_from_file(f_tanh)
#     # data_complex = get_InOut_from_file(f_complex)
#     # data_xor = get_InOut_from_file(f_xor)
#
#     #print(data_complex)
#
# if __name__ == "__main__":
#     main()