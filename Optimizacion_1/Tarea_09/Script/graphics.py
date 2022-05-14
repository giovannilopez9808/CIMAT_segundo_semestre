from Modules.functions import read_image, read_image_result
from Modules.methods import optimize_method
from Modules.params import get_params
import matplotlib.pyplot as plt
from os.path import join
from os import makedirs


params = get_params()
for lambda_value in params["lambda values"]:
    for method in params["methods"]:
        datasets = {
            "lambda": lambda_value,
            "tau": 1e-3,
            "GC method": method
        }
        params = get_params(datasets)
        optimize = optimize_method(params)
        folder = optimize.get_folder_results()
        filename = optimize.get_image_filename_results()
        filename = join(folder,
                        filename)
        image = read_image(params)
        image_result = read_image_result(filename)
        image = image.reshape(image_result.shape)
        fig, (ax1, ax2) = plt.subplots(1, 2,
                                       figsize=(8, 4))
        ax1.imshow(image,
                   cmap="gray")
        ax1.set_title("Original")
        ax1.axis("off")
        ax2.imshow(image_result,
                   cmap="gray")
        ax2.axis("off")
        ax2.set_title("Modificada")
        plt.tight_layout()
        folder = folder.replace(params["path results"],
                                params["path graphics"])
        makedirs(folder,
                 exist_ok=True)
        filename = filename.replace(params["path results"],
                                    params["path graphics"])
        filename = filename.replace("csv", "png")
        plt.savefig(filename)
