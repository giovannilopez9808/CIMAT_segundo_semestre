from os import makedirs as mkdir


def obtain_all_params() -> dict:
    """
    Funcion que reune los parámtros de la función y el gradiente. Devuelve estos dos parámetros en forma de diccionarios.
    """
    params = {
        "path data": "Data",
        "path results": "Results",
        "file data": "data.csv",
        "data column": "Max",
        "models": {
            "SGD": "#f72585",
            "NAG": "#3a0ca3",
            "ADAM": "#d00000",
            "ADADELTA": "#006400",
        },
        "n": 120,
        "max iteration": 200,
        "m": 18,
        "sigma": 4.8,
    }

    # parámetros del algoritmo
    gd_params = {
        'alpha': 1e-4,
        'alphaADADELTA': 1e-3,
        'alphaADAM': 0.95,
        'nIter': 300,
        'batch_size': 500,
        'eta': 0.9,
        'eta1': 0.9,
        'eta2': 0.999
    }
    mkdir(params["path results"],
          exist_ok=True)
    return params, gd_params
