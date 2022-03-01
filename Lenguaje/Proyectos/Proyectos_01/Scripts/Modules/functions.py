from sklearn.feature_selection import chi2, SelectKBest
from numpy import array
from os import listdir


def join_path(path: str, filename: str) -> str:
    """
    Une la direccion de un archivo con su nombre
    """
    return "{}{}".format(path, filename)


def obtain_name_place_from_filename(filename: str) -> str:
    """
    Obtiene el nombre del lugar en base a su nombre
    """
    name = filename.split(".")[0]
    name = name.split("_")
    name = " ".join(name)
    return name


def obtain_best_features(bow: array, labels: array, k: int = 1000) -> list:
    features = SelectKBest(chi2, k=k)
    features.fit(bow, labels)
    best_features = features.get_support(indices=True)
    return best_features, features.scores_


def obtain_target_matrix(index: dict, data: array, best_features: array) -> array:
    invert_index = {}
    for word in index:
        invert_index[index[word]] = word
    target_words = [invert_index[word] for word in best_features]
    target_matrix = array([data[index[word]] for word in target_words])
    return target_words, target_matrix


def ls(path: str) -> list:
    files = sorted(listdir(path))
    files = [file for file in files if "." in file]
    return files
