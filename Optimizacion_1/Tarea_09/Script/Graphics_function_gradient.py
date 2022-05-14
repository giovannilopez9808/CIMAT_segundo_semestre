from Modules.params import get_graphics_params, get_params
from Modules.methods import optimize_method
import matplotlib.pyplot as plt
from pandas import read_csv
from os.path import join


def get_ticks(params: dict, axis: str, graph: str = None) -> list:
    if axis == "x":
        lim = params[f"{axis} lim"]
        delta = params[f"{axis} delta"]
    if axis == "y":
        lim = params[f"{axis} lim"][graph]
        delta = params[f"{axis} delta"][graph]
    values = range(lim[0],
                   lim[1]+delta,
                   delta)
    return values


params = get_params()
graphics_params = get_graphics_params()
for lambda_value in params["lambda values"]:
    g_params = graphics_params[lambda_value]
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   figsize=(8, 5),
                                   sharex=True)
    for method in params["methods"]:
        color = params["methods"][method]
        datasets = {
            "lambda": lambda_value,
            "tau": 1e-3,
            "GC method": method
        }
        params = get_params(datasets)
        optimize = optimize_method(params)
        folder = optimize.get_folder_results()
        filename = optimize.get_iterations_filename_results()
        filename = join(folder,
                        filename)
        data = read_csv(filename,
                        index_col=0)
        ax1.plot(data.index,
                 data["Function"],
                 color=color,
                 label=method,
                 alpha=0.5,
                 ls="--")
        ax2.plot(data.index,
                 data["Gradient"],
                 color=color,
                 alpha=0.5,
                 ls="--")
    ax1.grid(ls="--",
             alpha=0.7,
             color="#000000")
    ax2.grid(ls="--",
             alpha=0.7,
             color="#000000")
    ax2.set_xlabel("Iteraciones")
    ax1.set_ylabel("$f(x)$")
    ax1.set_xlim(g_params["x lim"][0],
                 g_params["x lim"][1])
    ax1.set_xticks(get_ticks(g_params, "x"))
    ax1.set_yticks(get_ticks(g_params,
                             "y",
                             "Function"))
    ax2.set_yticks(get_ticks(g_params,
                             "y",
                             "Gradient"))
    ax2.set_ylabel("$\\nabla f(x)$")
    fig.legend(ncol=len(params["methods"]),
               frameon=False,
               loc="upper center")
    plt.tight_layout(pad=2)
    plt.savefig("test.png",
                dpi=400)
    break
