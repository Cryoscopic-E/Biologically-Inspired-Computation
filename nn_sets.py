class NNSets:
    """
    This class takes the array of inputs read from file and create:
    - A training set
    - A test set
    Using the data provided and split (by default) 70% to training set and 30% to test set.
    """

    def __init__(self, data):
        self.training_set = list()
        self.test_set = list()
        self.split_inputs(data)

    def split_inputs(self, data):
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
        self.inputs = inputs
        self.output = output

    def __repr__(self):
        return f"\nInputs : {self.inputs}\nOutput : {self.output}\n"

# sets = NNSets(d_inputs)
# print("TRAINING SET")
# print(sets.get_training_set())
# print("TEST SET")
# print(sets.get_test_set())
