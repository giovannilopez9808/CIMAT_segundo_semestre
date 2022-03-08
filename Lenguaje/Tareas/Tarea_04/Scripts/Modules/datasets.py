def obtain_parameters() -> dict:
    """
    Obtiene las rutas y nombres de los archivos que seran usados
    """
    parameters = {
        # Ruta de los archivos
        "path data": "Data/",
        "path graphics": "Graphics/",
        "path results": "Results/",
        # Archivos de entrenamiento
        "train": {
            "data": "mex_train.txt",
            "labels": "mex_train_labels.txt"
        },
        # Archivos de validacion
        "validation": {
            "data": "mex_val.txt",
            "labels": "mex_val_labels.txt"
        },
        "max words": 5000,
        "max bigrams": 1000,
        "lambda list": [[1/3, 1/3, 1/3],
                        [0.4, 0.4, 0.2],
                        [0.2, 0.4, 0.4],
                        [0.5, 0.4, 0.1],
                        [0.1, 0.4, 0.5]]
    }
    return parameters
