from operator import index
from functions import *


def clean_movies(data: pd.DataFrame, genres: list) -> pd.DataFrame:
    data_copy = data.copy()
    for index in data.index:
        check = 0
        for genre in genres:
            if genre in data["genres"][index]:
                check += 1
        # Al menos dos categorias
        if check < 2:
            data_copy = data_copy.drop(index)
    return data_copy


def clean_ratings(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    ratings_list = []
    movies_id = []
    for index in ratings.index:
        movie_id = ratings["movieId"][index]
        if movie_id in movies.index:
            rating = ratings["rating"][index]
            ratings_list += [rating]
            movies_id += [movie_id]
    return ratings_list, movies_id


def mean_movies(data: pd.DataFrame) -> pd.DataFrame:
    set_movies_id = set(data["movie_id"])
    data_set = pd.DataFrame(0.0, index=set_movies_id,
                            columns=["rating"])
    for movie_id in set_movies_id:
        data_movie = data[data["movie_id"] == movie_id]
        mean = data_movie["rating"].mean()
        data_set["rating"][movie_id] = mean
    data_set = data_set.round(4)
    return data_set


parameters = obtain_parameters()
parameters["file results"] = "select_rating.csv"
genres = obtain_select_genres()
movies = read_movie_list()
movies = split_genres(movies)
movies = clean_movies(movies, genres)
ratings = read_rating_list()
rating_list, movies_id = clean_ratings(movies, ratings)
data = pd.DataFrame(rating_list,
                    columns=["rating"])
data["movie_id"] = movies_id
data_set = mean_movies(data)
data_set.to_csv(join_path(parameters["path results"],
                          parameters["file results"]),
                index_label="Movies_id")
