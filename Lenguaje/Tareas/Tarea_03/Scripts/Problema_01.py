from Modules.functions import load_data, obatin_reduced_matrix, obtain_best_features, obtain_target_matrix
from Modules.graphics import create_constellation, create_centroid
from Modules.datasets import obtain_parameters
from Modules.dictionaries import dictionaries
from Modules.vocabulary import vocabularies
from Modules.DTR import build_TCOR
from Modules.bows import BoW

parameters = obtain_parameters()
vocabulary = vocabularies()
dictionary = dictionaries()
bows = BoW()
data_tr, labels_tr, data_val, labels_val = load_data(parameters)
vocabulary = vocabulary.obtain(data_tr, parameters["max words"])
index_word = dictionary.build_of_index(vocabulary)
bow_binary = bows.build_binary(data_tr, vocabulary, index_word)
tcor = build_TCOR(data_tr, vocabulary, index_word)
best_features = obtain_best_features(bow_binary, labels_tr)
target_words, target_matrix = obtain_target_matrix(index_word,
                                                   tcor,
                                                   best_features)
reduced_matrix = obatin_reduced_matrix(target_matrix)
create_constellation(reduced_matrix,
                     target_words,
                     parameters)
create_centroid(reduced_matrix,
                target_words,
                parameters)
