from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from Modules.vocabulary import vocabulary_class
from sklearn.preprocessing import normalize
from .tripadvisor import tripadvisor_model
from numpy import array, vectorize


class BoW:
    def __init__(self, vocabulary: vocabulary_class) -> None:
        self.vectorizer = CountVectorizer(stop_words=vocabulary.stopwords)

    def build_binary(self, tripadvisor: tripadvisor_model, word_index: dict) -> array:
        self.vectorizer.vocabulary = word_index.keys()
        self.result = self.vectorizer.fit_transform(
            tripadvisor.data["Opinión"])
        bow = normalize(self.result.toarray())
        return bow

    def build_TFIDF(self, tripadvisor: tripadvisor_model, word_index: dict, data_select: bool = False) -> array:
        vectorize = TfidfVectorizer(stop_words=self.vectorizer.stop_words,
                                    vocabulary=word_index.keys())
        if data_select:
            self.result = vectorize.fit_transform(
                tripadvisor.data_select["Opinión"])
        else:
            self.result = vectorize.fit_transform(tripadvisor.data["Opinión"])
        bow = normalize(self.result.toarray())
        return bow
