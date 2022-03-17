from Modules.functions import join_path
from Modules.plot_functions import mnist_image
from Modules.datasets import datasets_class
from Modules.mnist import mnist_model
import matplotlib.pyplot as plt
from random import choices
datasets = datasets_class()
parameters = datasets.obtain_dataset("log likelihood bisection")
mnist = mnist_model(parameters)
index = choices(range(len(mnist.train_data)),
                k=9)
images = []
for i in index:
    image = mnist_image(mnist.train_data, i)
    images += [image.copy()]
fig, axs = plt.subplots(3, 3, figsize=(9, 9))
axs = axs.flat
for i, ax in enumerate(axs):
    ax.imshow(images[i],
              cmap="gray")
    ax.axis("off")
plt.tight_layout()
plt.savefig(join_path(parameters["path graphics"],
                      "mnist.png"),
            dpi=400)
