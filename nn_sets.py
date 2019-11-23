import numpy as np


def get_in_out_from_file(txt_file):
    """
    Load inputs and outputs from txt file
    :param txt_file: text file name
    :return:
    """
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

    def __init__(self, data_file, split_percentage=70):
        self.training_set = list()
        self.test_set = list()
        data = get_in_out_from_file(data_file)
        self.split_ad_randomize_inputs(data, split_percentage)

    def split_ad_randomize_inputs(self, data, split_percentage, seed=19):
        """
        Randomize data position in array and split as training set and test set

        :param data: data from text file
        :param split_percentage: amount to split from original data array as training set (default 70%)
        :param seed: seed to use for consistency of data trough experiments (to use uncomment)
        :return:
        """
        # np.random.seed(seed)
        np.random.shuffle(data)
        inputs_length = len(data)
        training_length = int((split_percentage / 100.0) * inputs_length)
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
    """
    Set class holding inputs and respective outputs
    """

    def __init__(self, inputs, output):
        self.input = np.array(inputs)
        self.output = np.array(output)

    def __repr__(self):
        return f"\nInput : {self.input}\nOutput : {self.output}\n"
