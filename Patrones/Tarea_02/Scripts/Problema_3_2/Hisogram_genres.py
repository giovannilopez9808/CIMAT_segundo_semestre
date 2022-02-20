from functions import *

parameters = obtain_parameters()
parameters["file results"] = "Histogram_genres.csv"
movies = read_movie_list()
movies = split_genres(movies)
ratings = read_rating_list()
genres = obtain_all_genres(only_names=True)
data = pd.DataFrame(0,
                    index=genres,
                    columns=["Count"])
genres_list = []
for index in ratings.index:
    movie_id = ratings["movieId"][index]
    genres_movie = movies["genres"][movie_id]
    for genre in genres_movie:
        genres_list += [genre]
for genre in genres:
    count = genres_list.count(genre)
    data["Count"][genre] = count
data = data.sort_values("Count",
                        ascending=False)
data.to_csv(join_path(parameters["path results"],
                      parameters["file results"]))
