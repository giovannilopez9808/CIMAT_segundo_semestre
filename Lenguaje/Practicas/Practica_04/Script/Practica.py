from functions import obtain_best_features, obtain_target_matrix, obatin_reduce_matrix_sklearn
from functions import load_data, build_DOR
from datasets import obtain_parameters
from dictionaries import dictionaries
from vocabulary import vocabularies
from bows import BoW

parameters = obtain_parameters()
vocabulary = vocabularies()
dictionary = dictionaries()
bows = BoW()
data_tr, labels_tr, data_val, labels_val = load_data(parameters)
vocabulary = vocabulary.obtain(data_tr, parameters["max words"])
index_word = dictionary.build_of_index(vocabulary)
bow_binary = bows.build_binary(data_tr, vocabulary, index_word)
dor = build_DOR(bow_binary)
best_features = obtain_best_features(bow_binary, labels_tr)
target_matrix = obtain_target_matrix(index_word, dor, best_features)
# reduce_matrix = obatin_reduce_matrix(target_matrix)
# print("-"*40)
# print(reduce_matrix)
reduce_matrix_sklearn = obatin_reduce_matrix_sklearn(target_matrix)
print("-"*40)
print(reduce_matrix_sklearn)
