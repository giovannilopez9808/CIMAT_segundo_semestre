from Modules.functions import join_path, obtain_best_features, obtain_name_place_from_filename, ls
from Modules.tripadvisor import tripadvisor_model
from Modules.dictionary import dictionary_model
from Modules.vocabulary import vocabulary_class
from Modules.datasets import parameters_model
from Modules.BoW import BoW_class
from pandas import DataFrame

dataset = parameters_model()
dataset.parameters["path results"] += "K_best/"
tripadvisor = tripadvisor_model(dataset)
vocabulary_model = vocabulary_class()
dictionary = dictionary_model()
bow = BoW_class(vocabulary_model)
files = ls(dataset.parameters["path data"])
results = {}
result_basis = {"Words": [],
                "Scores": []}
for file in files:
    nameplace = obtain_name_place_from_filename(file)
    tripadvisor.read_data(file)
    vocabulary = vocabulary_model.obtain(tripadvisor)
    word_index = dictionary.build_word_index(vocabulary)
    index_word = dictionary.build_index_word(word_index)
    binary, words = bow.build_TFIDF(tripadvisor,
                                    word_index,
                                    return_words=True)
    best_features, scores = obtain_best_features(binary,
                                                 tripadvisor.data["new scale"])
    results = {}
    for index in best_features:
        results[words[index]] = scores[index]
    results = dictionary.sort_dict(results)
    results = DataFrame(results,
                        columns=["Words", "Scores"])
    filename = join_path(dataset.parameters["path results"],
                         file)

    results.to_csv(filename, index=False)
