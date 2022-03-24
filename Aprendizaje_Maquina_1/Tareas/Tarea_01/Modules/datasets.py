def obtain_all_params() -> dict:
    """
    Funcion que reune los parámtros de la función y el gradiente. Devuelve estos dos parámetros en forma de diccionarios.
    """
    params = {
        "models": ["SGD",
                   "NAG",
                   "ADAM",
                   "ADADELTA"],
        "max iteration": 100,
        "n": 100,
        "sigma": 1,
        "epsilon": 0.01
    }

    # parámetros del algoritmo
    gd_params = {
        'alpha': 0.95,
        'alphaADADELTA': 0.95,
        'alphaADAM': 0.95,
        'nIter': 300,
        'batch_size': 10,
        'eta': 0.9,
        'eta1': 0.9,
        'eta2': 0.999
    }
    return params, gd_params
