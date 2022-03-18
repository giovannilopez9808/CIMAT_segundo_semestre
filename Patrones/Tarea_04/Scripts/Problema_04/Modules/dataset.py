from Modules.models import KMeans_model_class
from numpy import array, reshape, shape
from .functions import join_path
import matplotlib.pyplot as plt
from pandas import DataFrame
from PIL import Image


def obtain_parms() -> dict:
    params = {"path data": "Data/",
              "path results": "Results/",
              "path graphics": "Graphics/",
              "file image": "Colorful-Flowers.jpg"}
    return params


class image_class:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.iterations = 5
        self.k_values = [2, 4, 8, 16, 32]
        self.read()

    def read(self) -> array:
        filename = join_path(self.params["path data"],
                             self.params["file image"])
        image = Image.open(filename)
        self.image = array(image)
        self.dim = shape(image)
        self.data = self.image.reshape(self.dim[0]*self.dim[1], 3)
        self.data = self.data/255

    def run_kmeans(self) -> dict:
        kmean_model = KMeans_model_class()
        results = {}
        for k in self.k_values:
            print("Resolviendo {} cluster".format(k))
            results[k] = {"scores": [],
                          "model": []}
            for i in range(self.iterations):
                model = kmean_model.run(k, self.data)
                results[k]["scores"] += [model.inertia_]
                results[k]["model"] += [model]
        self.results = results

    def compress(self) -> dict:
        self.images = {}
        for k in self.k_values:
            models = self.results[k]["model"]
            self.images[k] = {}
            for i, model in enumerate(models):
                image = model.cluster_centers_[model.predict(self.data)]
                image = reshape(image, (self.dim))
                image = image
                self.images[k][i] = image
        self.plot_images()

    def save(self) -> None:
        filename = join_path(self.params["path results"],
                             "kmean.csv")
        scores = {}
        for k in self.k_values:
            scores[k] = self.results[k]["scores"]
        scores = DataFrame(scores)
        scores.index.name = "Attempt"
        scores.to_csv(filename)

    def plot_images(self) -> None:
        for k in self.k_values:
            filename = "cluster_{}.png".format(k)
            filename = join_path(self.params["path graphics"],
                                 filename)
            images = self.images[k]
            fig, axs = plt.subplots(1, self.iterations,
                                    figsize=(18, 4))
            for i in range(self.iterations):
                ax = axs[i]
                ax.imshow(images[i])
                ax.axis("off")
            plt.tight_layout(pad=0.5)
            plt.savefig(filename,
                        bbox_inches="tight",
                        pad_inches=0,)
