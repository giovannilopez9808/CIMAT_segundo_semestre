from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def obtain_parameters():
    parameters = {"path data": "../../Data/Movielens/",
                  "path graphics": "../../Graphics/Problema_3_2/",
                  "path results": "../../Results/Problema_3_2/",
                  "Movie list": "movies.csv",
                  "Rating list": "ratings.csv",
                  "Tags list": "tags.csv",
                  "Link list": "links.csv"}
    return parameters


def obtain_all_genres(only_names: bool = False) -> list:
    genders = ['Crime',
               'Animation',
               'Action',
               'Sci-Fi',
               'IMAX',
               'Children',
               'Comedy',
               'Adventure',
               'Western',
               'Horror',
               'Musical',
               'Thriller',
               'Fantasy',
               'War',
               '(no genres listed)',
               'Documentary',
               'Film-Noir',
               'Mystery',
               'Drama',
               'Romance']
    colors = obtain_colors()
    if only_names:
        return genders
    genders_dict = {}
    for gender, color in zip(genders, colors):
        genders_dict[gender] = color
    return genders_dict

def obtain_select_genres(only_names: bool = True):
    genres = [
        "Drama",
        "Comedy",
        "Action",
        # "Thriller",
        # "Adventure"
    ]
    if only_names:
        return genres
    colors = [
        "#a4133c",
        "#03045e",
        "#f72585",
        "#06d6a0",
        "#dc2f02"
    ]
    genders_dict = {}
    for gender, color in zip(genres, colors):
        genders_dict[gender] = color
    return genders_dict


def obtain_colors():
    colors = [
        '#e6194B',
        '#3cb44b',
        '#ffe119',
        '#dcbeff',
        '#4363d8',
        '#f58231',
        '#911eb4',
        '#42d4f4',
        '#f032e6',
        '#bfef45',
        '#fabed4',
        '#469990',
        '#9A6324',
        '#fffac8',
        '#800000',
        '#aaffc3',
        '#808000',
        '#ffd8b1',
        '#000075',
        '#a9a9a9'
    ]
    return colors


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


def read_movie_list():
    parameters = obtain_parameters()
    data = read_data(parameters["path data"],
                     parameters["Movie list"],
                     use_index=True)
    return data


def read_rating_list():
    parameters = obtain_parameters()
    data = read_data(parameters["path data"],
                     parameters["Rating list"],
                     use_index=False)
    return data


def obtain_all_scores():
    labels = np.linspace(0.5, 5, 10)
    return labels


def split_genres(data: pd.DataFrame) -> pd.DataFrame:
    data["genres"] = data["genres"].astype("str").str.split("|")
    return data


def assign_genre(movies: pd.DataFrame, data: pd.DataFrame) -> pd.DataFrame:
    data["genre"] = ""
    data["colors"] = ""
    genres = obtain_select_genres(only_names=False)
    for movie_id in data.index:
        genres_list = movies["genres"][movie_id]
        for genre in genres_list:
            if genre in genres:
                data.loc[movie_id, "genres"] = genre
                data.loc[movie_id, "colors"] = genres[genre]
                break
    return data


def plot_MDS_results(parameters: dict):
    movies = read_movie_list()
    genres = obtain_select_genres()
    movies = split_genres(movies)
    data = read_data(parameters["path results"],
                     parameters["file data"],
                     use_index=True)
    data = assign_genre(movies, data)
    for genre in genres:
        data_genre = data[data["genres"] == genre]
        if len(data_genre):
            plt.scatter(data_genre["x"],
                        data_genre["y"],
                        c=data_genre["colors"],
                        label=genre,
                        alpha=0.2)
    plt.axis("off")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(join_path(parameters["path graphics"],
                          parameters["file graphics"]))


def euclidean(vector: np.array) -> np.array:
    n = len(vector)
    matrix = np.zeros((n, n))
    for i in range(n):
        v_i = vector[i]
        for j in range(i, n):
            v_j = vector[j]
            matrix[i, j] += (v_i-v_j)**2
            matrix[j, i] += (v_i-v_j)**2
    return matrix


def gaussian(vector: np.array, sigma: float) -> np.array:
    n = len(vector)
    matrix = np.zeros((n, n))
    for i in range(n):
        v_i = vector[i]
        for j in range(i, n):
            v_j = vector[j]
            matrix[i, j] = np.exp(-(v_i-v_j)**2/sigma)
            matrix[j, i] = np.exp(-(v_i-v_j)**2/sigma)
    return matrix


def linear(vector: np.array) -> np.array:
    matrix = np.outer(vector, vector)
    return matrix


def sigmod(vector: np.array, sigma: float, c: float) -> np.array:
    n = len(vector)
    matrix = np.zeros((n, n))
    for i in range(n):
        v_i = vector[i]
        for j in range(i, n):
            v_j = vector[j]
            matrix[i, j] = np.exp(-(v_i-v_j)**2/sigma+c)
            matrix[j, i] = np.exp(-(v_i-v_j)**2/sigma+c)
    return matrix


def obtain_vector_of_rating(parameters: dict):
    data = read_data(parameters["path results"],
                     parameters["file data"],
                     use_index=True)
    vector = list(data["rating"])
    return data, vector


def run_MDS_model(data, matrix, parameters):
    results = pd.DataFrame(index=data.index,
                           columns=["x", "y"])
    model2d = MDS(random_state=42,
                  dissimilarity='precomputed')
    data_model = model2d.fit_transform(matrix)
    results_matrix = model2d.embedding_
    results["x"] = results_matrix[:, 0]
    results["y"] = results_matrix[:, 1]
    results.to_csv(join_path(parameters["path results"],
                             parameters["file results"]),
                   index_label="rating")
