from numpy import array, concatenate, logical_or, zeros
from .functions import join_path
import pickle
import gzip


class mnist_model:
    def __init__(self, parameters: dict) -> None:
        """
        Modelo de lectura y organizacion de los datos de Mnist
        ---------------------
        Inputs:
        + parameters -> diccionario con los parametros de ruta y nombre de archivo de datos
        """
        self.parameters = parameters
        self.read()

    def read(self):
        """
        Lectura de los datos de entrenamiento, validacion y test a partir de un archivo
        """
        # Obtiene el nombre y direccion del archivo
        filename = join_path(self.parameters["path data"],
                             self.parameters["file data"])
        with gzip.open(filename, 'rb') as file:
            u = pickle._Unpickler(file)
            u.encoding = 'latin1'
            train, val, test = u.load()
        # Split de los datos
        self.train_data, self.train_label = self.clean_data(train)
        self.val_data, self.val_label = self.clean_data(val)
        self.test_data, self.test_label = self.clean_data(test)

    def clean_data(self, data: array) -> tuple:
        """
        Limpieza de los datos de entrenamiento, validacion y test. Obtiene unicamente los datos que sus etiquetas sean 0 y 1
        ---------------------
        Inputs:
        + data -> array con los datos de entrenamiento y sus etiquetas

        ---------------------
        Output:
        + vectors -> matriz de las imagenes
        + lables -> etiquetas de 0 y 1 de cada imagen
        """
        index = logical_or(data[1] == 1, data[1] == 0)
        vectors = data[0][index]
        labels = data[1][index]
        vectors_aumented = zeros((vectors.shape[0],
                                  vectors.shape[1] + 1))
        for i in range(vectors.shape[0]):
            vectors_aumented[i] = concatenate((vectors[i], [1.0]))
        return vectors_aumented, labels
