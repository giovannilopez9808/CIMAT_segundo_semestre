from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def obtain_parameters() -> dict:
    parameters = {"path data": "../Data/",
                  "path graphics": "../Graphics/",
                  "path graphics animales": "../Graphics/Animals/",
                  "matrix file": "predicate-matrix-continuous.txt",
                  "classes file": "classes.txt"}
    return parameters


def join_path(path: str, filename: str) -> str:
    return "{}{}".format(path, filename)


def plot_image(path: str, filename: str, ax: plt.subplot, position: list) -> None:
    x, y = position
    filename = join_path(path, filename)
    image = mpimg.imread(filename)
    imagebox = OffsetImage(image, zoom=0.05)
    ab = AnnotationBbox(imagebox, (x, y),
                        frameon=False)
    ax.add_artist(ab)
