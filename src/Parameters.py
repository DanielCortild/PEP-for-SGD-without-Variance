import numpy as np

class ParameterList:
    """
    A class for storing and retrieving information about a given parameter
    """
    UPPER_BOUND = 1000

    def __init__(self, value=np.zeros(1), name="", description=""):
        if not isinstance(value, np.ndarray):  # being stubborn here
            raise ValueError(f"Parameter must me instantiated with a np.ndarray.")
        else:
            self.value = value
            self.name = name
            self.description = description

    def __repr__(self):
        return f"ParameterList({self.name}, {self.value})"

    def __str__(self):
        return f"ParameterList({self.name}, {self.value})"

    def average(self):
        # returns the average of the parameter
        return np.mean(self.value)

    def max(self):
        # returns the max of the parameter
        return np.max(self.value)

    def min(self):
        # returns the min of the parameter
        return np.min(self.value)

    def is_bounded(self):
        # returns True if the parameter is bounded
        return self.max() < self.UPPER_BOUND

class Parameter:
    """
    A class for storing and retrieving information about a given parameter
    """

    def __init__(self, value=0.0, name="", description=""):
        self.value = value
        self.name = name
        self.description = description

    def __repr__(self):
        return f"Parameter({self.name}, {self.value})"

    def __str__(self):
        return f"Parameter({self.name}, {self.value})"

class BagOfParameters:
    """
    Will contain all the parameters. Basically a fancy dictionary
    """

    def __init__(self):
        pass