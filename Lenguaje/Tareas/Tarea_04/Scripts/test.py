from Modules.datasets import obtain_parameters
from Modules.tweets import tweets_data

parameters = obtain_parameters()
tweets = tweets_data(parameters)
tweets.obtain_perplexity()
