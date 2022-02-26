from Modules.DTR import obtain_cosine_similitud, write_top_similitud_documents
from Modules.datasets import obtain_parameters
from Modules.dictionaries import dictionaries
from Modules.vocabulary import vocabularies
from Modules.functions import load_data
from Modules.bows import BoW

parameters = obtain_parameters()
vocabulary = vocabularies()
dictionary = dictionaries()
bows = BoW()
data_tr, labels_tr, data_val, labels_val = load_data(parameters)
vocabulary = vocabulary.obtain(data_tr, parameters["max words"])
index_word = dictionary.build_of_index(vocabulary)
word_index = dictionary.obtain_invert_dictionary(index_word)
bow = bows.build_tfidf(data_tr, vocabulary, index_word)
distances = obtain_cosine_similitud(bow)
write_top_similitud_documents(distances,
                              data_tr,
                              parameters)
