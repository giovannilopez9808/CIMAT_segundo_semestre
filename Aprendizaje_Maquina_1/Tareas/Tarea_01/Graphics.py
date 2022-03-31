from Modules.datasets import obtain_all_params
from Modules.functions import function_class
from pandas import read_csv, to_datetime
from numpy import array, linspace
import matplotlib.pyplot as plt
from os.path import join

params, gd_params = obtain_all_params()
function = function_class()
filename = join(params["path data"],
                params["file data"])
data = read_csv(filename,
                index_col=0,
                parse_dates=[0])
data = data.dropna(axis=0)
data = data[params["data column"]]
fig, axs = plt.subplots(2, 2,
                        sharex=True,
                        sharey=True,
                        figsize=(18, 7))
axs = axs.flatten()
for ax, model_name in zip(axs, params["models"]):
    filename = "{}.csv".format(model_name)
    filename = join(params["path results"],
                    filename)
    data_model = read_csv(filename)
    alpha = array(data_model["alpha"])
    mu = array(data_model["mu"])
    x = linspace(1, len(data), len(data))
    phi = function.phi(x, mu, params["sigma"])
    ax.plot(data.index,
            phi@alpha,
            label=model_name,
            ls="--",
            color=params["models"][model_name])
    ax.plot(data.index,
            data,
            label="Data",
            color="#0b525b")
ax.set_xlim(to_datetime("2000-01-01"),
            to_datetime("2020-01-01"))
ax.set_ylim(4, 14)
# plt.legend()
lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
lines, labels = [sum(label, []) for label in zip(*lines_labels)]
by_label = dict(zip(labels, lines))
fig.legend(by_label.values(),
           by_label.keys(),
           frameon=False,
           ncol=5,
           bbox_to_anchor=(0.67, 1))
plt.tight_layout(pad=2)
plt.savefig("test.png")
