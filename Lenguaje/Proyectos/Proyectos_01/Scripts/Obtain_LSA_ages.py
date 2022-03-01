from Modules.functions import join_path, obtain_name_place_from_filename, ls
from Modules.tripadvisor import tripadvisor_model
from Modules.dictionary import dictionary_model
from Modules.vocabulary import vocabulary_class
from Modules.datasets import parameters_model
from Modules.LSA import LSA
from Modules.BoW import BoW

vocabulary_model = vocabulary_class()
dictionary = dictionary_model()
dataset = parameters_model()
dataset.parameters["path results"] += "LSA/Ages/"
bow = BoW(vocabulary_model)
tripadvisor = tripadvisor_model(dataset)
ages = ["Joven", "Mayor"]
files = ls(dataset.parameters["path data"])
for file in files:
    print("Analizando {}".format(file))
    nameplace = obtain_name_place_from_filename(file)
    tripadvisor.read_data(file)
    for age in ages:
        path_results = "{}{}/".format(dataset.parameters["path results"],
                                      age)
        tripadvisor.select_data_per_age_range(age)
        vocabulary = vocabulary_model.obtain(tripadvisor,
                                             data_select=True)
        word_index = dictionary.build_word_index(vocabulary)
        tfidf = bow.build_TFIDF(tripadvisor,
                                word_index,
                                data_select=True)
        lsa = LSA(tfidf, word_index, 3)
        lsa.obtain_words()
        filename = join_path(path_results,
                             file)
        lsa.words.to_csv(filename)
