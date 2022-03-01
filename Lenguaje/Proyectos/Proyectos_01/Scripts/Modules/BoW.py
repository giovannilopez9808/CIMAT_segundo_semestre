from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from Modules.vocabulary import vocabulary_class
from sklearn.preprocessing import normalize
from .tripadvisor import tripadvisor_model
from numpy import array, vectorize
from pandas import DataFrame


class BoW:
    def __init__(self, vocabulary: vocabulary_class) -> None:
        self.vectorizer = CountVectorizer(stop_words=vocabulary.stopwords)

    def build_binary(self, tripadvisor: tripadvisor_model, word_index: dict, data_select: bool = False, ngram_range: tuple = (1, 1), return_words: bool = False) -> array:
        vectorizer = CountVectorizer(stop_words=self.vectorizer.stop_words,
                                     vocabulary=word_index.keys(),
                                     ngram_range=ngram_range)
        data = self.select_data(tripadvisor,
                                data_select)
        bow = self.build(data, vectorizer)
        if return_words:
            return bow, vectorizer.get_feature_names_out()
        else:
            return bow

    def build_TFIDF(self, tripadvisor: tripadvisor_model, word_index: dict, data_select: bool = False, ngram_range: tuple = (1, 1), return_words: bool = False) -> array:
        vectorizer = TfidfVectorizer(stop_words=self.vectorizer.stop_words,
                                     #  vocabulary=word_index.keys(),
                                     ngram_range=ngram_range)
        data = self.select_data(tripadvisor,
                                data_select)
        bow = self.build(data, vectorizer)
        if return_words:
            return bow, vectorizer.get_feature_names_out()
        else:
            return bow

    def select_data(self, tripadvisor: tripadvisor_model, data_select: bool):
        if data_select:
            data = tripadvisor.data_select["Opinión"]
        else:
            data = tripadvisor.data["Opinión"]
        return data

    def build(self, data: DataFrame, vectorizer: vectorize):
        self.result = vectorizer.fit_transform(data)
        bow = normalize(self.result.toarray())
        return bow
