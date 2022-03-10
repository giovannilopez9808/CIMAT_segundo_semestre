from .functions import join_path
from numpy import loadtxt


class data_class:
    """
    Lectura y organización de base de datos de animals with attributes
    ------
    Inputs:
    + parameters -> diccionario con los parametros de la localizacion y nombre de cada archivo. Estos pueden ser modificados en la funcion obtain_parameters
    """

    def __init__(self, parameters: dict) -> None:
        self.parameters = parameters
        self.read()

    def read(self):
        """
        Lectura de la matriz de atributos y nombres de los animales
        """
        self.read_matrix()
        self.read_classes()
        self.len = len(self.colors)

    def read_matrix(self):
        """
        Lectura de la matriz de atributos
        """
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
