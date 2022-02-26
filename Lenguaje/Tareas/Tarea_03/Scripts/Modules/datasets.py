def obtain_parameters() -> dict:
    """
    Obtiene las rutas y nombres de los archivos que seran usados
    """
    parameters = {
        # Ruta de los archivos
        "path data": "../Data/",
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
        # Archivo de EmoLex
        "EmoLex": "emolex.txt",
        # Archivo de SEL
        "SEL": "SEL.txt",
        "max words": 5000,
        "max bigrams": 1000,
        "path graphics": "../Graphics/",
        "constellation name": "constellation.png",
        "centroid name": "centroid.png",
        "subset words":  ["tristes", "triste", "alegría", "hermosa", "chica", "hombres", "hdp", "madre", "madres", "@usuario", "hijos", "pendeja", "pendejo", "mierda", "loca", "hijo", "hija", "mamá", "tía"]
    }
    return parameters
