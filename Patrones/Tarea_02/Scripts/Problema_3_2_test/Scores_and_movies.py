from functions import *


def clean_data(data: pd.DataFrame, genres: list) -> pd.DataFrame:
    data_copy = data.copy()
    for index in data.index:
        check = 0
        for genre in genres:
            if genre in data["genres"][index]:
                check += 1
        # Al menos una categoria categorias
        if check < 2:
            data_copy = data_copy.drop(index)
    return data_copy


def fill_data(movies: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    users_id = set(ratings["userId"])
    data = pd.DataFrame(0.0, columns=movies.index,
                        index=users_id)
    for index in ratings.index:
        movie_id = ratings["movieId"][index]
        if movie_id in movies.index:
            rating = ratings["rating"][index]
            user_id = ratings["userId"][index]
            data[movie_id][user_id] = rating
    return data


parameters = obtain_parameters()
parameters["file results"] = "Test/ratings_select_movies.csv"
genres = obtain_select_genres()
movies = read_movie_list()
movies = split_genres(movies)
movies = clean_data(movies, genres)
ratings = read_rating_list()
data = fill_data(movies,
                 ratings)
data.to_csv(join_path(parameters["path results"],
                      parameters["file results"]))
