"""
Clase que contiene la estructura para la ejecuccion del modelo de PCA para diferentes datos dado el número de componentes a obtener y el nombre del kernel
"""

from numpy import array, linspace, meshgrid, min, max, vstack
from .params import get_colors_array
from pandas import DataFrame
from sklearn.svm import SVC


class SVM_model:
    def __init__(self, params: dict) -> None:
        self.params = params

    def create(self, kernel: str) -> None:
        """
        Inicializa el modelo PCA dado el numero de componentes y el nombre del kernel
        """
        self.model = SVC(C=10,
                         kernel=kernel,)

    def run(self, data: array, labels: array) -> None:
        """
        Ejecuta el modelo PCA dado un embedding
        """
        self.model.fit(data, labels)
        self.get_grid(data)

    def get_grid(self, data: array) -> array:
        x, y = data.T.copy()
        x = linspace(min(x),
                     max(x),
                     50)
        y = linspace(min(y),
                     max(y),
                     50)
        y, x = meshgrid(y, x)
        self.grid = vstack([x.ravel(),
                            y.ravel()]).T
        self.grid_predict = self.preditct(self.grid)
        self.get_predict_colors()
        self.grid = self.grid.T

    def get_suport_vectors(self) -> DataFrame:
        """
        Obtiene los eigenvectores de los resultados de PCA
        """
        results = self.model.support_vectors_
        results = results.T
        return results

    def get_predict_colors(self) -> array:
        self.colors = get_colors_array(self.params,
                                       self.grid_predict)

    def preditct(self, data: array) -> array:
        results = self.model.predict(data)
        return results
