from pandas import DataFrame, read_csv


def join_path(path: str, filename: str) -> str:
    """
    Creacion de la ruta donde se encuentra un archivo
    """
    return "{}{}".format(path, filename)


def obtain_filename(parameters: dict) -> str:
    """
    Obtiene el nombre del archivo en base a el dataset dado
    """
    filename = "lambda_{}.csv".format(parameters["lambda"])
    if parameters["test data"]:
        filename = "lambda_{}_test.csv".format(parameters["lambda"])
    return filename


def read_data(filename: str) -> DataFrame:
    """
    Lectura estandarizada de los datos
    """
    data = read_csv(filename, index_col=0)
    return data


def write_status(name: str) -> None:
    """
    Impresion estandarizada
    """
    text = "Resolviendo {}".format(name)
    print("\n")
    print("-"*len(text))
    print(text)


def obtain_graphics_filename(parameters: dict) -> str:
    """
    Obtiene el nombre del archivo en base a el dataset dado
    """
    filename = "lambda_{}.png".format(parameters["lambda"])
    if parameters["test data"]:
        filename = "lambda_{}_test.png".format(parameters["lambda"])
    return filename


def obtain_colors_per_method() -> dict:
    colors = {"x": "#007200",
              "y": "#9d0208",
              }
    return colors
