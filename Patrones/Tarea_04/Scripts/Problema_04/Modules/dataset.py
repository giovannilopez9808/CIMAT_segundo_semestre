from xml.dom import minicompat

from torch import minimum
from Modules.models import KMeans_model_class
from numpy import array, reshape, shape
from .functions import join_path
import matplotlib.pyplot as plt
from pandas import DataFrame
from PIL import Image


def obtain_parms() -> dict:
    """
    Rutas y nombres de archivos a utilizar
    """
    params = {"path data": "Data/",
              "path results": "Results/",
              "path graphics": "Graphics/",
              "file image": "Colorful-Flowers.jpg"}
    return params


class image_class:
    def __init__(self, params: dict) -> None:
        """
        Modelo de la imagen que realiza su lectura, organzación y ejecuccion de los modelos de kmeans
        """
        self.params = params
        self.iterations = 5
        self.k_values = [2, 4, 8, 16, 32]
        self.init = ["k-means++",
                     "random",
                     "array"]
        self.read()

    def read(self) -> array:
        """
        Lectura y formateo de la imagen
        """
        filename = join_path(self.params["path data"],
                             self.params["file image"])
        image = Image.open(filename)
        # Lectura de la imagen
        self.image = array(image)
        # Dimensiones en pixeles de la imagen
        self.dim = shape(image)
        # Flat pixels
        self.data = self.image.reshape(self.dim[0]*self.dim[1], 3)
        # Normalizacion
        self.data = self.data/255

    def run_kmeans(self) -> dict:
        """
        Ejecuccion del modelo de kmeans para los diferentes inicializadores
        """
        kmean_model = KMeans_model_class()
        results = {}
        for init in self.init:
            print("Resolviendo para el init {}".format(init))
            results[init] = {}
            for k in self.k_values:
                results[init][k] = {}
                for i in range(self.iterations):
                    results[init][k][i] = {}
                    model = kmean_model.run(k, self.data, init)
                    results[init][k][i]["score"] = model.inertia_
                    results[init][k][i]["model"] = model
        self.results = results

    def compress(self) -> dict:
        """
        Compresion de las imagenes usando el modelo que tenga el mínimo valor de score
        """
        self.images = {}
        for k in self.k_values:
            self.images[k] = {}
            for init in self.init:
                models = self.results[init][k]
                minimum_model = min(models.items(),
                                    key=lambda x: x[0])
                minimum_model = minimum_model[1]
                model = minimum_model["model"]
                image = model.cluster_centers_[model.predict(self.data)]
                image = reshape(image, (self.dim))
                image = image
                self.images[k][init] = image
        self.plot_images()

    def save(self) -> None:
        """
        Guardado de los resultados obtenidos
        """
        for init in self.init:
            filename = init.replace("-", "_")
            filename = "{}.csv".format(filename)
            filename = join_path(self.params["path results"],
                                 filename)
            scores = {}
            for k in self.k_values:
                scores[k] = []
                for i in range(self.iterations):
                    scores[k] += [self.results[init][k][i]["score"]]
            scores = DataFrame(scores)
            scores.index.name = "Attempt"
            scores.to_csv(filename)

    def plot_images(self) -> None:
        """
        Ploteo de las imagenes de cada inicializador obtenido. Genera una imagen por número de cluster usado
        """
        for k in self.k_values:
            filename = "cluster_{}.png".format(k)
            filename = join_path(self.params["path graphics"],
                                 filename)
            images = self.images[k]
            fig, axs = plt.subplots(1, len(self.init),
                                    figsize=(18, 4))
            for i, init in enumerate(self.init):
                ax = axs[i]
                ax.set_title("init = {}".format(init))
                ax.imshow(images[init])
                ax.axis("off")
            plt.tight_layout(pad=0.5)
            plt.savefig(filename,
                        bbox_inches="tight",
                        pad_inches=0,)
