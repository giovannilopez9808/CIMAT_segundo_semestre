from functions import format_mnist, join_path, obtain_sets_isomap, read_data
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from pandas import DataFrame
from numpy import dot

parameters = {
    "path data": "../../Data/",
    "mnist data": "mnist_test.csv",
    "path graphics": "../../Graphics/Problema_3_1/",
    "file graphics": "isomap.png"
}

data = read_data(parameters["path data"], parameters["mnist data"])
data = format_mnist(data)
matrix = dot(data.T, data)
isomap = Isomap(n_neighbors=3, n_components=3)
isomap.fit(matrix)
results = isomap.transform(matrix)
results = DataFrame(results,
                    columns=["Component 1", "Component 2", "Component 3"])
fig = plt.figure(figsize=(14, 6))
ax = fig.add_subplot(121, projection="3d")
ax.scatter(results["Component 1"],
           results["Component 2"],
           results["Component 3"],
           marker=".",
           color="#000000",
           alpha=0.25)
subset = obtain_sets_isomap(False)
for index in subset:
    ax.scatter(results["Component 1"][index],
               results["Component 2"][index],
               results["Component 3"][index],
               marker="o",
               s=50,
               color=subset[index])
    ax.text(results["Component 1"][index], results["Component 2"][index],
            results["Component 3"][index], str(index), "y")
ax.view_init(45, -164)
ax.grid()
ax = fig.add_subplot(122, projection="3d")
ax.scatter(results["Component 1"],
           results["Component 2"],
           results["Component 3"],
           marker=".",
           color="#000000",
           alpha=0.25)
subset = obtain_sets_isomap(False)
for index in subset:
    ax.scatter(results["Component 1"][index],
               results["Component 2"][index],
               results["Component 3"][index],
               marker="o",
               s=50,
               color=subset[index])
    ax.text(results["Component 1"][index],
            results["Component 2"][index],
            results["Component 3"][index],
            str(index),
            "x",
            color="#000000")
ax.view_init(17, -128)
ax.grid()
plt.savefig(join_path(parameters["path graphics"],
                      parameters["file graphics"]),
            dpi=800)
