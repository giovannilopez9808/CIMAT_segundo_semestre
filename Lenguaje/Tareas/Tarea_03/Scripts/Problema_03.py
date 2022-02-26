from Modules.functions import load_data, print_results
from Modules.datasets import obtain_parameters
from Modules.dictionaries import dictionaries
from Modules.vocabulary import vocabularies
from Modules.models import SVM_model
from Modules.DTR import build_TCOR, tcor_to_BoW
from Modules.bows import BoW
from numpy import shape
parameters = obtain_parameters()
vocabulary = vocabularies()
dictionary = dictionaries()
model = SVM_model()
bows = BoW()
results = []
data_tr, labels_tr, data_val, labels_val = load_data(parameters)
vocabulary = vocabulary.obtain(data_tr, parameters["max words"])
index_word = dictionary.build_of_index(vocabulary)
tcor_tr = build_TCOR(data_tr, vocabulary, index_word)
bow_tr = tcor_to_BoW(data_tr, vocabulary, index_word, tcor_tr)
del tcor_tr
tcor_val = build_TCOR(data_val, vocabulary, index_word)
bow_val = tcor_to_BoW(data_val, vocabulary, index_word, tcor_val)
del tcor_val
grid = model.create_model(bow_tr, labels_tr)
result = model.evaluate_model(bow_val, labels_val, grid, "TCOR")
results += [result]
del grid
bow_tr = bows.build_tfidf(data_tr, vocabulary, index_word)
bow_val = bows.build_tfidf(data_val, vocabulary, index_word)
grid = model.create_model(bow_tr, labels_tr)
result = model.evaluate_model(bow_val, labels_val, grid, "TFIDF")
results += [result]
print_results(results)
