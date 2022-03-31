def obtain_all_params() -> dict:
    """
    Funcion que reune los parámtros de la función y el gradiente. Devuelve estos dos parámetros en forma de diccionarios.
    """
    params = {
        "models": [
            # "SGD",
            "NAG",
            # "ADAM",
            # "ADADELTA",
        ],
        "n": 120,
        "max iteration": 10,
        "m": 20,
        "sigma": 200,
    }

    # parámetros del algoritmo
    gd_params = {
        'alpha': 1e-3,
        'alphaADADELTA': 1e-3,
        'alphaADAM': 0.95,
        'nIter': 300,
        'batch_size': 500,
        'eta': 0.9,
        'eta1': 0.9,
        'eta2': 0.999
    }
    return params, gd_params
