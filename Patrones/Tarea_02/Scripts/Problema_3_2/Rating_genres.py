from functions import *

parameters = obtain_parameters()
parameters["file result"] = "Rating_genre.csv"
ratings = read_rating_list()
movies = read_movie_list()
movies = split_genres(movies)
genres_list = obtain_all_genres(only_names=True)
scores = obtain_all_scores()
data = pd.DataFrame(0,
                    columns=genres_list,
                    index=scores)
for index in ratings.index:
    movie_id = ratings["movieId"][index]
    genres = movies["genres"][movie_id]
    for genre in genres:
        rating = ratings["rating"][index]
        data[genre][rating] += 1
data.to_csv(join_path(parameters["path results"],
                      parameters["file result"]))
