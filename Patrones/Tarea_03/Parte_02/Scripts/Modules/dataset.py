from .functions import join_path
from pandas import DataFrame
from numpy import loadtxt


class data_class:
    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.read()

    def read(self):
        self.read_matrix()
        self.read_classes()
        self.len = len(self.colors)

    def read_matrix(self):
        filename = join_path(self.parameters["path data"],
                             self.parameters["matrix file"])
        data = loadtxt(filename)
        self.matrix = data

    def read_classes(self):
        filename = join_path(self.parameters["path data"],
                             self.parameters["classes file"])
        colors, data = loadtxt(filename,
                               unpack=True,
                               dtype=str)
        self.names = data
        self.colors = colors
