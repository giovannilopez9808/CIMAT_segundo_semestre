from Modules.functions import concat_dataframes, ls, obtain_best_features
from Modules.tripadvisor import tripadvisor_model
from Modules.vocabulary import vocabulary_class
from Modules.dictionary import dictionary_model
from Modules.datasets import parameters_model
from Modules.BoW import BoW_class
from pandas import DataFrame


filename = "distribution_nationality.csv"
dataset = parameters_model()
dictionary = dictionary_model()
vocabulary_model = vocabulary_class()
bow_model = BoW_class(vocabulary_model)
tripadvisor = tripadvisor_model(dataset)
files = ls(dataset.parameters["path data"])
data = DataFrame()
# Concatenate all data
for file in files:
    tripadvisor.read_data(file)
    data = concat_dataframes(data, tripadvisor.data.copy())
tripadvisor.data = data.copy()
del data
tripadvisor.sort_by_date()
vocabulary = vocabulary_model.obtain(tripadvisor)
word_index = dictionary.build_word_index(vocabulary)
index_word = dictionary.build_index_word(word_index)
bow, words = bow_model.build_TFIDF(tripadvisor,
                                   word_index,
                                   return_words=True)
best_features, scores = obtain_best_features(bow,
                                             tripadvisor.data["new scale"],
                                             10)
words_list = []
for index in best_features:
    words_list += [words[index]]
for file in files:
    print(file)
    tripadvisor.read_data(file)
    tripadvisor.sort_by_date()
    opinios = tripadvisor.obtain_opinions_as_text(vocabulary_model.stopwords)
    opinios.dispersion_plot(words_list)
