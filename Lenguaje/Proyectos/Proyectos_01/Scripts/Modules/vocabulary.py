from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import TweetTokenizer as tokenizer
from .tripadvisor import tripadvisor_model
from nltk import bigrams as bigrams_nltk
from nltk.corpus import stopwords
from nltk import FreqDist
from numpy import array


class vocabulary_class:
    def __init__(self, max_words: int = 5000) -> None:
        self.vectorizer = CountVectorizer()
        self.tokenizer = tokenizer().tokenize
        self.max_words = max_words
        self.stopwords = self.obtain_stopwords()

    def obtain_stopwords(self):
        stopwords_list = stopwords.words("spanish")
        stopwords_list += stopwords.words("english")
        stopwords_list += [".", ",", "...", "!", "(", ")", "¡", "-", ":"]
        return set(stopwords_list)

    def obtain(self, tripadvisor: tripadvisor_model, data_select: bool = False) -> list:
        """
        Obtiene la lista de una distribucion de frecuencias de palabras ordenada de mayor a menor a partir de una lista de oraciones
        """
        # Inicializacion de la lista que guardara los tokens
        corpus = []
        if data_select:
            data = tripadvisor.data_select
        else:
            data = tripadvisor.data
        for oration in data["Opinión"]:
            tokens = self.tokenizer(oration)
            tokens = [
                token for token in tokens if not token in self.stopwords]
            corpus += tokens
        # Creacion de la distribucion de frecuencias
        vocabylary = FreqDist(corpus)
        vocabylary = self.sort_freqdist(vocabylary)
        vocabylary = self.split_data(vocabylary)
        return vocabylary

    def sort_freqdist(self, vocabylary: FreqDist) -> list:
        """
        Ordena la lista de distribucion de frecuencias de palabras de mayor frecuencia a menor
        """
        aux = [(vocabylary[key], key) for key in vocabylary]
        aux.sort()
        aux.reverse()
        return aux

    def split_data(self, data: list) -> list:
        """
        Realiza la separacion de elementos en una lista dado el numero de elementos que se quieren conservar
        """
        return data[:self.max_words]
