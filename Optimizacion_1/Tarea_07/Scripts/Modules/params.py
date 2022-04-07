from numpy import array, ones, pi
from os.path import join
from os import makedirs


def get_datasets() -> dict:
    datasets = {
        "iteration": 30,
        "step models": [
            "newton",
            "dogleg",
            "cauchy"
        ],
        "functions": [
            "wood",
            "rosembrock",
            "branin"
        ]
    }
    return datasets


def get_params() -> dict:
    params = {
        "path results": "../Results",
        "path graphics": "../Graphics",
        'tau vector': 1e-6,
        'tau function': 1e-6,
        'tau gradient': 1e-6,
        'delta k': 0.1,
        'delta max': 0.2,
        'eta': 0.2,
        'eta min': 0.25,
        'eta max': 0.75,
        'eta1 hat': 0.25,
        'eta2 hat': 2.0
    }
    mkdir(params["path results"])
    mkdir(params["path graphics"])
    return params


def get_function_params(function_name: str) -> dict:
    params = {
        "wood": {
            "n": 4,
            "optimal position": ones(4),
        },
        "rosembrock": {
            "n": 100,
            "optimal position": ones(100),
        },
        "branin": {
            "n": 2,
            "optimal position": array([pi, 2.275]),
            "a": 1,
            "b": 5.1/(4*pi**2),
            "c": 5/pi,
            "r": 6,
            "s": 10,
            "t": 1/(8*pi),
        }
    }
    return params[function_name]


def mkdir(path: str) -> None:
    makedirs(path,
             exist_ok=True)


def obtain_path(params: dict, dataset: dict) -> str:
    path = join(params["path results"],
                dataset["function"],
                dataset["step model"])
    return path


def obtain_filename_iteration(params: dict, dataset: dict, iteration: int) -> str:
    n = str(iteration).zfill(2)
    filename = "{}.csv".format(n)
    path = obtain_path(params,
                       dataset)
    path = join(path,
                "Iteration")
    mkdir(path)
    filename = join(path, filename)
    return filename
