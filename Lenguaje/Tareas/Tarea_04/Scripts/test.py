from Modules.datasets import obtain_parameters
from Modules.tweets import tweets_data

parameters = obtain_parameters()
tweets = tweets_data(parameters)
print("-"*40)
print("Datos de prueba")
tweets.obtain_perplexity(use_data_test=True)
print("-"*40)
print("Datos de validación")
tweets.obtain_perplexity(use_data_test=False)
