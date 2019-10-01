from os import path


def read_file_input(txt_file):
    f=open(txt_file, "r")
    lines = f.readlines()

    for line in lines:
        print(line)

# def main():
#      script_dir = path.dirname(__file__)
#      filepath = path.abspath(path.join(script_dir,"Data/1in_cubic.txt"))
#      read_file_input(filepath)
#
# if __name__ == "__main__":
#     main()