from Modules.DTR import build_TCOR, obtain_cosine_similitud, write_top_similitud_words
from Modules.datasets import obtain_parameters
from Modules.dictionaries import dictionaries
from Modules.vocabulary import vocabularies
from Modules.functions import load_data
from Modules.bows import BoW
from tabulate import tabulate

parameters = obtain_parameters()
vocabulary = vocabularies()
dictionary = dictionaries()
bows = BoW()
data_tr, labels_tr, data_val, labels_val = load_data(parameters)
vocabulary = vocabulary.obtain(data_tr, parameters["max words"])
index_word = dictionary.build_of_index(vocabulary)
word_index = dictionary.obtain_invert_dictionary(index_word)
bow_binary = bows.build_binary(data_tr, vocabulary, index_word)
tcor = build_TCOR(data_tr, vocabulary, index_word, weight="PPMI")
distances = obtain_cosine_similitud(tcor)
write_top_similitud_words(distances, word_index)
