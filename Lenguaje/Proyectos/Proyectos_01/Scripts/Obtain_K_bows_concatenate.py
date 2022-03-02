from Modules.functions import join_path, obtain_name_place_from_filename, ls, obtain_best_features
from Modules.tripadvisor import tripadvisor_model
from Modules.dictionary import dictionary_model
from Modules.vocabulary import vocabulary_class
from Modules.datasets import parameters_model
from Modules.BoW import BoW_class
from numpy import concatenate
from pandas import DataFrame

vocabulary_model = vocabulary_class()
dictionary = dictionary_model()
dataset = parameters_model()
dataset.parameters["path results"] += "BoW_concatenate/"
bow = BoW_class(vocabulary_model)
tripadvisor = tripadvisor_model(dataset)
files = ls(dataset.parameters["path data"])
for file in files:
    nameplace = obtain_name_place_from_filename(file)
    tripadvisor.read_data(file)
    vocabulary_model.max_words = 1000
    vocabulary_1000 = vocabulary_model.obtain(tripadvisor)
    vocabulary_model.max_words = 2000
    vocabulary_2000 = vocabulary_model.obtain(tripadvisor)
    word_index_1000 = dictionary.build_word_index(vocabulary_1000)
    word_index_2000 = dictionary.build_word_index(vocabulary_2000)
    unigram, words_unigram = bow.build_TFIDF(tripadvisor,
                                             word_index_1000,
                                             return_words=True)
    bigram, words_bigram = bow.build_TFIDF(tripadvisor,
                                           word_index_2000,
                                           ngram_range=(2, 2),
                                           return_words=True)
    trigram, words_trigram = bow.build_TFIDF(tripadvisor,
                                             word_index_1000,
                                             ngram_range=(3, 3),
                                             return_words=True)
    bows = concatenate((unigram,
                       bigram,
                       trigram),
                       axis=1)
    words = concatenate((words_unigram,
                        words_bigram,
                        words_trigram))
    best_features, scores = obtain_best_features(bows,
                                                 tripadvisor.data["new scale"],
                                                 1000)
    results = {}
    for index in best_features:
        results[words[index]] = scores[index]
    results = dictionary.sort_dict(results)
    results = DataFrame(results,
                        columns=["Words", "Scores"])
    filename = join_path(dataset.parameters["path results"],
                         file)
    filename = join_path(dataset.parameters["path results"],
                         file)
    results.to_csv(filename,
                   index=False)
