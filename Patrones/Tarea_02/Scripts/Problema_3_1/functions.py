import matplotlib.pyplot as plt
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


def obtain_sets(use_flat: bool = True) -> list:
    sets = np.array([[303, 6733, 2802],
                    [4958, 126, 5786],
                    [2589, 60, 5338]])
    if use_flat:
        return sets.flatten()
    return sets
