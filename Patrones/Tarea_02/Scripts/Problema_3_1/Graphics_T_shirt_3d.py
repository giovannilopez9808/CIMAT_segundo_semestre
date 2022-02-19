import matplotlib.pyplot as plt
from functions import *

parameters = {"path data": "../../Data/",
              "mnist data": "mnist_test.csv",
              "path graphics": "../../Graphics/Problema_3_1/",
              "file graphics": "T_shirts_3d.png"}

data = read_data(parameters["path data"],
                 parameters["mnist data"])
data = format_mnist(data)
sets = obtain_sets_3d()
fig, axs = plt.subplots(1, 3, figsize=(10, 10))
axs = np.array(axs).flatten()
for i, index in enumerate(sets):
    vector_mnist = data[index]
    image_mnist = vector_image_to_matrix_image(vector_mnist)
    axs[i].axis("off")
    axs[i].imshow(image_mnist,
                  cmap="Greys")
    axs[i].set_title("Imagen {}".format(i+1),
                     fontsize=14)
plt.tight_layout()
plt.savefig("{}{}".format(parameters["path graphics"],
                          parameters["file graphics"]),
            dpi=400)
