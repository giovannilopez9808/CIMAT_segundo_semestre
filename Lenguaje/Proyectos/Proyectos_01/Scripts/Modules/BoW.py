from sklearn.feature_extraction.text import CountVectorizer
from Modules.vocabulary import vocabulary_class
from sklearn.preprocessing import normalize
from .tripadvisor import tripadvisor_model
from numpy import array


class BoW:
    def __init__(self, vocabulary: vocabulary_class) -> None:
        self.vectorizer = CountVectorizer(stop_words=vocabulary.stopwords)

    def build_binary(self, tripadvisor: tripadvisor_model) -> array:
        self.result = self.vectorizer.fit_transform(
            tripadvisor.data["Opinión"])
        bow = normalize(self.result.toarray())
        return bow
