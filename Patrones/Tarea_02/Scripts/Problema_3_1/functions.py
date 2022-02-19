import pandas as pd
import numpy as np


def vector_image_to_matrix_image(vector: list) -> np.array:
    matrix = np.reshape(list(vector), (28, 28))
    return matrix


def join_path(path: str, filename: str) -> str:
    path = "{}{}".format(path,
                         filename)
    return path


def read_data(path: str, filename: str, use_index: bool = False) -> pd.DataFrame:
    file = "{}{}".format(path,
                         filename)
    if use_index:
        data = pd.read_csv(file,
                           index_col=0)
    else:
        data = pd.read_csv(file)
    return data


def format_mnist(data: pd.DataFrame) -> pd.DataFrame:
    data.index = np.arange(1, len(data)+1)
    data = data[data["label"] == 0]
    data = data.drop(columns="label")
    data = data.transpose()
    return data


def obtain_sets_2d(only_numbers: bool = True) -> np.array:
    sets = {303: "#084c61",
            6733: "#bd1e1e",
            2802: "#b96913",
            67: "#ffc233",
            9686: "#ccff66",
            5786: "#e01a4f",
            2589: "#240046",
            60: "#008000",
            5338: "#f72585"}
    if only_numbers:
        return sets.keys()
    return sets


def obtain_sets_3d(only_numbers: bool = False) -> np.array:
    sets = {8681: "#084c61",
            39: "#bd1e1e",
            2877: "#b96913"}
    if only_numbers:
        return sets.keys()
    return sets
