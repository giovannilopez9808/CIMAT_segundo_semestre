from matplotlib.pyplot import show
from pyparsing import restOfLine
from Modules.functions import join_path, obtain_best_features, obtain_name_place_from_filename, ls
from Modules.tripadvisor import tripadvisor_model
from Modules.dictionary import dictionary_model
from Modules.vocabulary import vocabulary_class
from Modules.datasets import parameters_model
from Modules.Graphics import plot_word_cloud
from pandas import DataFrame
from Modules.BoW import BoW

dataset = parameters_model()
dataset.parameters["path results"] += "K_best/Negatives/"
tripadvisor = tripadvisor_model(dataset)
vocabulary_model = vocabulary_class()
dictionary = dictionary_model()
bow = BoW(vocabulary_model)
files = ls(dataset.parameters["path data"])
for file in files:
    print(file)
    nameplace = obtain_name_place_from_filename(file)
    tripadvisor.read_data(file)
    tripadvisor.obtain_only_negatives_scores()
    vocabulary = vocabulary_model.obtain(tripadvisor,
                                         data_select=True)
    word_index = dictionary.build_word_index(vocabulary)
    index_word = dictionary.build_index_word(word_index)
    bow_tfidf = bow.build_TFIDF(tripadvisor,
                                word_index,
                                data_select=True)
    best_features, scores = obtain_best_features(bow_tfidf,
                                                 tripadvisor.data_select["Escala"],
                                                 'all')
    results = {}
    for index in best_features:
        results[index_word[index]] = scores[index]
    results = dictionary.sort_dict(results)
    results = DataFrame(results,
                        columns=["Words", "Scores"])
    filename = join_path(dataset.parameters["path results"],
                         file)
    results.to_csv(filename, index=False)
