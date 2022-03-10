from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from .functions import join_path
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def plot_image(path: str, filename: str, ax: plt.subplot, position: list) -> None:
    x, y = position
    filename = join_path(path, filename)
    image = mpimg.imread(filename)
    imagebox = OffsetImage(image, zoom=0.05)
    ab = AnnotationBbox(imagebox, (x, y),
                        frameon=False)
    ax.add_artist(ab)


def plot(model, names, parameters: dict, include_images: bool = True, alpha: float = 0.5, color: str = "#370617") -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(model.results["Component 1"],
               model.results["Component 2"],
               alpha=alpha,
               c=color)
    if include_images:
        for index in model.animals_index:
            animal_image = "{}.jpg".format(names[index])
            position = [model.results["Component 1"][index],
                        model.results["Component 2"][index]]
            plot_image(parameters["path graphics animales"],
                       animal_image,
                       ax,
                       position
                       )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(join_path(parameters["path graphics"],
                          parameters["file graphics"]))


def plot_clusters(parameters: dict, cluster_model, model, data):
    fig, axs = plt.subplots(1, 3,
                            sharey=True,
                            figsize=(18, 6))
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
                          parameters["file graphics"]))
