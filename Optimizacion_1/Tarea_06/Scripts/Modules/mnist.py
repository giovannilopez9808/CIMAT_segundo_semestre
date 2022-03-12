from numpy import array, concatenate
from .functions import join_path
from numpy.linalg import norm
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
        # Apertura del archivo
        file = gzip.open(filename, "rb")
        # Codificacion y lectura de los datos
        u = pickle._Unpickler(file)
        u.encoding = "latin1"
        train, val, test = u.load()
        file.close()
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
        vectors = []
        labels = []
        matrix = data[0]
        vector = data[1]
        n = len(vector)
        for i in range(n):
            if vector[i] in [0, 1]:
                vectors += [matrix[i]]
                labels += [vector[i]]
        vectors = vectors/norm(vectors, axis=1)[:, None]
        vectors_copy = []
        for i in range(len(vectors)):
            vectors_copy += [concatenate((vectors[i], [1.0]))]
        vectors_copy = array(vectors_copy)
        labels = array(labels)
        return vectors_copy, labels
