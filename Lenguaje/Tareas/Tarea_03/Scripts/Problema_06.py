from Modules.functions import load_data, obtain_information_gain, print_tuple_as_table, sort_dict
from Modules.datasets import obtain_parameters
from Modules.dictionaries import dictionaries
from Modules.graphics import create_wordcloud
from Modules.vocabulary import vocabularies

parameters = obtain_parameters()
vocabulary = vocabularies()
dictionary = dictionaries()
data_tr, labels_tr, data_val, labels_val = load_data(parameters)
vocabulary = vocabulary.obtain(data_tr, parameters["max words"])
index_word = dictionary.build_of_index(vocabulary)
ig = obtain_information_gain(data_tr,
                             labels_tr,
                             data_val,
                             labels_val,
                             index_word)
create_wordcloud(ig, parameters)
ig = sort_dict(ig)
print_tuple_as_table(ig, 50)
