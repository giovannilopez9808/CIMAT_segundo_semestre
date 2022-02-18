from functions import *

parameters = obtain_parameters()
data_tr, labels_tr, data_val, labels_val = load_data(parameters)
vocabulary = obtain_vocabylary(data_tr, parameters["max words"])
index_word = create_dictonary_of_index(vocabulary)
bow_binary = build_binary_bow(data_tr, vocabulary, index_word)
dor = build_DOR(bow_binary)
best_features = obtain_best_features(bow_binary, labels_tr)
target_matrix = obtain_target_matrix(index_word, dor, best_features)
reduce_matrix = obatin_reduce_matrix(target_matrix)
