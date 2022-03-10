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


def plot(model, names, parameters: dict) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(model.results["Component 1"],
               model.results["Component 2"],
               alpha=0.5,
               color="#370617")
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
