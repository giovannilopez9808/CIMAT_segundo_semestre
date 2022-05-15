from Modules.functions import read_image, read_image_result
from Modules.methods import optimize_method
from Modules.params import get_params
import matplotlib.pyplot as plt
from os.path import join
from os import makedirs
from numpy import sqrt


params = get_params()
for method in params["methods"]:
    fig, axs = plt.subplots(1, 5,
                            figsize=(16, 4))
    image = read_image(params)
    shape = int(sqrt(image.shape[0]))
    image = image.reshape((shape, shape))
    axs = axs.flatten()
    axs[0].imshow(image,
                  cmap="gray")
    axs[0].set_title("Original")
    axs[0].axis("off")
    for i, lambda_value in enumerate(params["lambda values"]):
        datasets = {
            "lambda": lambda_value,
            "tau": 1e-3,
            "GC method": method
        }
        ax = axs[i+1]
        params = get_params(datasets)
        optimize = optimize_method(params)
        folder = optimize.get_folder_results()
        filename = optimize.get_image_filename_results()
        filename = join(folder,
                        filename)
        image_result = read_image_result(filename)
        ax.imshow(image_result,
                  cmap="gray")
        ax.axis("off")
        ax.set_title(f"$\\lambda = {lambda_value}$")
        plt.tight_layout()
    filename = f"{method}.png"
    filename = join(params["path graphics"],
                    filename)
    plt.subplots_adjust(wspace=0.01,
                        hspace=0)
    plt.tick_params(pad=0)
    plt.savefig(filename,
                dpi=500,
                bbox_inches="tight",
                pad_inches=0.1)
