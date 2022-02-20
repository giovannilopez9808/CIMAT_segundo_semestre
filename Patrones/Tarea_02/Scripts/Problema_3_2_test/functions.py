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
        "#03045e",
        "#f72585",
        "#06d6a0",
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


def obatin_users_id():
    data = read_rating_list()
    users_id = data.index
    users_id = set(users_id)
    return users_id


def obtain_all_scores():
    labels = np.linspace(0.5, 5, 10)
    return labels


def split_genres(data: pd.DataFrame) -> pd.DataFrame:
    data["genres"] = data["genres"].astype("str").str.split("|")
    return data
