import numpy as np


class Matrix:
    def __init__(self, data=None, size=None):
        if data is not None:
            self.data = np.array(data)
        elif size is not None:
            self.data = np.random.rand(size, size).astype(np.float64)
        else:
            self.data = None

    def get_data_as_array(self):
        return self.data.copy()

    def set_data_from_array(self, array):
        self.data = np.array(array)

    def __eq__(self, other):
        return np.array_equal(self.data, other.data)

    def __str__(self):
        return str(self.data) if self.data is not None else "Empty Matrix"
