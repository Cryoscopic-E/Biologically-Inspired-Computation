import os
import os.path


def read_file_input(txt_file):
    f=open(txt_file, "r")
    lines = f.readlines()

    for line in lines:
        print(line)


def main():
    root_path = os.path.dirname(os.path.abspath(__file__))
    input_path = os.path.join(root_path, "Data", "1in_cubic.txt")
    read_file_input(input_path)


if __name__ == "__main__":
    main()