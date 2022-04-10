from pandas import DataFrame, read_csv
from Mex_data import Mex_data_class
from numpy.random import normal
from argparse import Namespace
from os.path import join
from numpy import array


class word2vec_class:
    """
    Modelo que lee y crea la matriz de embeddings de word2vec
    """

    def __init__(self, params: dict, args: Namespace) -> None:
        self.params = params
        self.args = args
        self.read()

    def read(self) -> None:
        """
        lectura de los embeddings de word2vec
        """
        filename = join(self.params["path word2vec"],
                        self.params["file word2vec"])
        data = read_csv(filename,
                        sep=" ",
                        skiprows=1,
                        header=0,
                        index_col=0)
        data = data.T
        self.vector_size = len(data)
        self.data = self.to_dict(data)

    def to_dict(self, data: DataFrame) -> dict:
        """
        creacion de un diccionario, las keys son las palabras de los diccionario y su valor es el vector de representacion
        """
        return data.to_dict("list")

    def obtain_embedding_matrix(self, Mex_data: Mex_data_class) -> array:
        """
        crea la matriz de embeddings de word2vec
        """
        matrix = normal(0, 1, size=(
            self.args.max_vocabulary, self.vector_size))
        for word in Mex_data.word_index:
            if word in self.data:
                id = Mex_data.word_index[word]
                matrix[id] = array(self.data[word], dtype=float)
        return matrix
