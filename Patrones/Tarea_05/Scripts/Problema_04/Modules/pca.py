"""
Clase que contiene la estructura para la ejecuccion del modelo de PCA para diferentes datos dado el número de componentes a obtener y el nombre del kernel
"""

from sklearn.decomposition import KernelPCA
from pandas import DataFrame
from numpy import array


class PCA_model:
    def __init__(self) -> None:
        pass

    def create(self, n_components: int, kernel: str) -> None:
        """
        Inicializa el modelo PCA dado el numero de componentes y el nombre del kernel
        """
        self.model = KernelPCA(n_components,
                               kernel=kernel,
                               )
        self.generate_header_names(n_components, kernel)

    def run(self, data: array) -> None:
        """
        Ejecuta el modelo PCA dado un embedding
        """
        self.model.fit(data)

    def generate_header_names(self, n_components: int, kernel: str) -> array:
        """
        Genera el nombre de los headers para guardar los datos
        """
        name = "Component {}"
        self.names = [name.format(i+1)
                      for i in range(n_components)]

    def get_eigenvectors(self) -> DataFrame:
        """
        Obtiene los eigenvectores de los resultados de PCA
        """
        results = DataFrame(self.model.eigenvectors_,
                            columns=self.names)
        return results
