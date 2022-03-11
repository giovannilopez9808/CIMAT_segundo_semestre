from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from scipy.cluster.hierarchy import dendrogram, linkage
from .functions import join_path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def plot_image(path: str, filename: str, ax: plt.subplot,
               position: list) -> None:
    """
    Realiza el ploteo de una imagen sobre una grafica
    ------------
    Inputs:
    path -> direccion de la carpeta donde se encuentran las fotos de los animales
    filename -> nombre de la foto del animal a graficar
    ax -> subplot donde se graficara
    position -> posicion donde se graficara la imagen
    """
    x, y = position
    filename_graphics = join_path(path, filename)
    image = mpimg.imread(filename_graphics)
    imagebox = OffsetImage(image, zoom=0.05)
    ab = AnnotationBbox(imagebox, (x, y), frameon=False)
    ax.add_artist(ab)


def plot(model,
         names,
         parameters: dict,
         include_images: bool = True,
         alpha: float = 0.5,
         color: str = "#370617") -> None:
    """
    Realiza el ploteo del resultado de la reduccion dimension junto a su color, ya sea proporcionado por el clustering o un color fijo
    --------------
    Inputs:
    model -> modelo de reduccion de dimension
    names -> nombre de de los animales a plotear
    parametes -> diccionario con los parametros de nombres y rutas
    alpha -> transparencia de los puntos
    color -> string o lista de colores
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(model.results["Component 1"],
               model.results["Component 2"],
               alpha=alpha,
               s=50,
               c=color)
    if include_images:
        for index in model.animals_index:
            animal_image = "{}.jpg".format(names[index])
            position = [
                model.results["Component 1"][index],
                model.results["Component 2"][index]
            ]
            plot_image(parameters["path graphics animales"], animal_image, ax,
                       position)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(join_path(parameters["path graphics"],
                          parameters["file graphics"]),
                dpi=400)


def plot_clusters(parameters: dict, cluster_model, model, data):
    """
    Realiza el ploteo de cada tipo de clusters
    ------------
    Inputs:
    parametes -> diccionario con los parametros de nombres y rutas
    cluster_model -> modelo con los metodos de clistering
    model -> modelo de reduccion de orden
    data -> clase de la informacion de animals with attributes
    """
    fig, axs = plt.subplots(1, 3, sharey=True, figsize=(18, 6))
    for ax, linkage in zip(axs, parameters["linkage types"]):
        cluster_model.run(data.matrix, 4, linkage)
        ax.set_title("Linkage {}".format(linkage))
        ax.scatter(model.results["Component 1"],
                   model.results["Component 2"],
                   alpha=1,
                   c=cluster_model.results)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(join_path(parameters["path graphics"],
                          parameters["file graphics"]),
                dpi=400)


def plot_denograma(parameters: dict, data) -> None:
    """
    Realiza el ploteo de los denogramas para cada tipo de clustering
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, linkage_name in zip(axs, parameters["linkage types"]):
        linked = linkage(data.matrix, linkage_name)
        ax.set_title("Linkage {}".format(linkage_name))
        dendrogram(linked,
                   orientation='top',
                   labels=data.names,
                   distance_sort='descending',
                   show_leaf_counts=True,
                   ax=ax)
        ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(
        join_path(parameters["path graphics"], parameters["file graphics"]))
