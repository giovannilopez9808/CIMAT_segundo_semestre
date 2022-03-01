from Modules.functions import join_path, obtain_best_features, obtain_name_place_from_filename
from Modules.dictionary import dictionary_model
from Modules.vocabulary import vocabulary_class
from Modules.datasets import parameters_model
from Modules.tripadvisor import tripadvisor_model
from pandas import DataFrame
from os import listdir as ls
from Modules.BoW import BoW

dataset = parameters_model()
dataset.parameters["path results"] += "K_best/"
tripadvisor = tripadvisor_model(dataset)
vocabulary_model = vocabulary_class()
dictionary = dictionary_model()
bow = BoW(vocabulary_model)
files = sorted(ls(dataset.parameters["path data"]))
results = {}
result_basis = {"Words": [],
                "Scores": []}
for file in files:
    nameplace = obtain_name_place_from_filename(file)
    tripadvisor.read_data(file)
    vocabulary = vocabulary_model.obtain(tripadvisor)
    word_index = dictionary.build_word_index(vocabulary)
    index_word = dictionary.build_index_word(word_index)
    binary = bow.build_binary(tripadvisor, word_index)
    best_features, scores = obtain_best_features(binary,
                                                 tripadvisor.data["new scala"])
    results = {}
    for index in best_features:
        results[index_word[index]] = scores[index]
    results = dictionary.sort_dict(results)
    results = DataFrame(results,
                        columns=["Words", "Scores"])
    filename = join_path(dataset.parameters["path results"],
                         file)

    results.to_csv(filename, index=False)
