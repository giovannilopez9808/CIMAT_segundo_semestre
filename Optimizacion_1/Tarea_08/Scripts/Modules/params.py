from numpy.random import sample


def get_params() -> dict:
    params = {
        "path data": "/home/giovanni/CIMAT_segundo_semestre/Optimizacion_1/Tarea_08/Scripts/Data",
        "path graphics": "/home/giovanni/CIMAT_segundo_semestre/Optimizacion_1/Tarea_08/Scripts/Graphics",
        "path results": "/home/giovanni/CIMAT_segundo_semestre/Optimizacion_1/Tarea_08/Scripts/Results",
        "Images": {"flower",
                   "grave",
                   "memorial",
                   "person1",
                   "rose",
                   "sheep"
                   }
    }
    return params


def get_f_params(dimension: int) -> dict:
    f_params = {"sigma": 0.5,
                "alpha": sample(dimension),
                "mu": sample((dimension, 3)),
                "search name": "bisection",
                "max iterations": 300,
                "c1": 1e-4,
                "c2": 0.9,
                "tau": 1e-6,
                }
    return f_params
