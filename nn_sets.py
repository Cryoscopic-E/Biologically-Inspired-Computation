import numpy as np


def get_in_out_from_file(txt_file):
    f = open(txt_file, "r")
    lines = f.readlines()
    data = []
    for line in lines:
        row_split = line.split()
        data.append(row_split)
    return np.array(data, dtype=float)


class NNSets:
    """
    This class takes the array of inputs read from file and create:
    - A training set
    - A test set
    Using the data provided and split (by default) 70% to training set and 30% to test set.
    """

    def __init__(self, data_file):
        self.training_set = list()
        self.test_set = list()
        data = get_in_out_from_file(data_file)
        self.split_ad_randomize_inputs(data)

    def split_ad_randomize_inputs(self, data, seed=19):
        np.random.seed(seed)
        np.random.shuffle(data)
        inputs_length = len(data)
        training_length = int(.7 * inputs_length)
        for _input in data:
            _i = _input[:len(_input) - 1] if len(_input) > 2 else _input[0]
            _o = _input[len(_input) - 1]
            if len(self.training_set) < training_length:
                self.training_set.append(_IOSet(_i, _o))
            else:
                self.test_set.append(_IOSet(_i, _o))

    def get_training_set(self):
        return self.training_set

    def get_test_set(self):
        return self.test_set


class _IOSet:

    def __init__(self, inputs, output):
        self.input = np.array(inputs)
        self.output = np.array(output)

    def __repr__(self):
        return f"\nInput : {self.input}\nOutput : {self.output}\n"
