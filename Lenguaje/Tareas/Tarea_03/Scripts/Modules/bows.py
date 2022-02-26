from nltk.tokenize import TweetTokenizer as tokenizer
from sklearn.preprocessing import normalize
from nltk import bigrams as bigrams_nltk
from .dictionaries import dictionaries
from numpy import zeros, array, log10
from nltk import FreqDist


class BoW:
    def __init__(self) -> None:
        pass

    def build_binary(self, data: list, vocabylary: list, index: dict) -> array:
        """
        Creacion de la BoW usando pesos binarios
        """
        bow = zeros((len(data), len(vocabylary)), dtype=float)
        docs = 0
        for tweet in data:
            vocabylary_data = FreqDist(tokenizer().tokenize(tweet))
            for word in vocabylary_data:
                if word in index.keys():
                    bow[docs, index[word]] = 1
            docs += 1
        return bow

    def build_binary_with_probabilities(data: list, vocabylary: list, index: dict,
                                        probability: dict) -> array:
        """
        Creacion de la BoW usando pesos binarios
        """
        bow = zeros((len(data), len(vocabylary)), dtype=float)
        docs = 0
        for tweet in data:
            vocabylary_data = FreqDist(tokenizer().tokenize(tweet))
            for word in vocabylary_data:
                if word in index.keys():
                    bow[docs, index[word]] = 1
                    if word in probability:
                        bow[docs, index[word]] = probability[word]
            docs += 1
        return bow

    def build_binary_bigram(self, data: list, vocabylary: list, index: dict) -> array:
        """
        Creacion de la BoW usando pesos binarios
        """
        bow = zeros((len(data), len(vocabylary)), dtype=float)
        docs = 0
        for tweet in data:
            bigrams = bigrams_nltk(tokenizer().tokenize(tweet))
            vocabylary_data = FreqDist(bigrams)
            for bigram in vocabylary_data:
                if bigram in index.keys():
                    bow[docs, index[bigram]] = 1
            docs += 1
        return bow

    def build_frecuency(self, data: list, vocabylary: list, index: dict) -> array:
        """
        Creacion de la BoW usando pesos basado en frecuencias
        """
        bow = zeros((len(data), len(vocabylary)), dtype=float)
        docs = 0
        for tweet in data:
            vocabylary_data = FreqDist(tokenizer().tokenize(tweet))
            for word in vocabylary_data:
                if word in index.keys():
                    bow[docs, index[word]] = vocabylary[word]
            docs += 1
        return bow

    def build_frecuency_with_probabilities(data: list, vocabylary: list,
                                           index: dict,
                                           probability: dict) -> array:
        """
        Creacion de la BoW usando pesos basado en frecuencias
        """
        bow = zeros((len(data), len(vocabylary)), dtype=float)
        docs = 0
        for tweet in data:
            vocabylary_data = FreqDist(tokenizer().tokenize(tweet))
            for word in vocabylary_data:
                if word in index.keys():
                    bow[docs, index[word]] = tweet.count(word)
                    if word in probability:
                        bow[docs, index[word]] *= probability[word]
            docs += 1
        return bow

    def build_frecuency_bigram(self, data: list, vocabylary: list,
                               index: dict) -> array:
        """
        Creacion de la BoW usando pesos basado en frecuencias
        """
        bow = zeros((len(data), len(vocabylary)), dtype=float)
        docs = 0
        for tweet in data:
            bigrams = bigrams_nltk(tokenizer().tokenize(tweet))
            vocabylary_data = FreqDist(bigrams)
            for bigram in vocabylary_data:
                if bigram in index.keys():
                    bow[docs, index[bigram]] = log10(
                        vocabylary_data[bigram] + 1)
            docs += 1
        return bow

    def build_tfidf(self, data: list, vocabylary: list, index: dict) -> array:
        """
        Creacion de la BoW usando pesos basado en frecuencias
        """
        # Inicilizacion del bow
        dictionary = dictionaries()
        bow = zeros((len(data), len(vocabylary)), dtype=float)
        # Total de oraciones
        n = len(data)
        # Inicializacion del diccionario que contiene la repeticion de cada palabra
        idf_per_word_and_document = dictionary.build_with_words_and_documents(index.keys(),
                                                                              data)
        for docs, tweet in enumerate(data):
            # Frecuencias
            vocabylary_data = FreqDist(tokenizer().tokenize(tweet))
            for word in vocabylary_data:
                if word in index.keys():
                    # Descriptiva
                    tf = tweet.count(word)
                    idf_per_word_and_document[word][docs] += 1
                    bow[docs, index[word]] = log10(tf + 1)
        # Discriminativa
        for word in index.keys():
            idf = sum(idf_per_word_and_document[word].values())
            idf = log10(n / idf)
            for docs, tweet in enumerate(data):
                bow[docs, index[word]] = bow[docs, index[word]] * idf
        bow = normalize(bow)
        return bow

    def build_tfidf_with_probabilities(self, data: list, vocabylary: list, index: dict,
                                       probability: dict) -> array:
        """
        Creacion de la BoW usando pesos basado en frecuencias
        """
        # Inicilizacion del bow
        dictionary = dictionaries()
        bow = zeros((len(data), len(vocabylary)), dtype=float)
        # Total de oraciones
        n = len(data)
        # Inicializacion del diccionario que contiene la repeticion de cada palabra
        idf_per_word_and_document = dictionary.build_with_words_and_documents(index.keys(),
                                                                              data)
        for docs, tweet in enumerate(data):
            # Frecuencias
            vocabylary_data = FreqDist(tokenizer().tokenize(tweet))
            for word in vocabylary_data:
                if word in index.keys():
                    # Descriptiva
                    tf = tweet.count(word)
                    idf_per_word_and_document[word][docs] += 1
                    bow[docs, index[word]] = log10(tf + 1)

        # Discriminativa
        for word in index.keys():
            idf = sum(idf_per_word_and_document[word].values())
            idf = log10(n / idf)
            for docs, tweet in enumerate(data):
                if word in probability:
                    bow[docs, index[word]] *= probability[word]
                bow[docs, index[word]] *= idf
        return bow

    def build_tfidf_bigram(self, data: list, vocabylary: list, index: dict) -> array:
        """
        Creacion de la BoW usando pesos basado en frecuencias
        """
        # Inicilizacion del bow
        dictionary = dictionaries()
        bow = zeros((len(data), len(vocabylary)), dtype=float)
        # Total de oraciones
        n = len(data)
        # Inicializacion del diccionario que contiene la repeticion de cada palabra
        idf_per_word_and_document = dictionary.build_with_words_and_documents(index.keys(),
                                                                              data)
        for docs, tweet in enumerate(data):
            # Frecuencias
            bigrams = bigrams_nltk(tokenizer().tokenize(tweet))
            vocabylary_data = FreqDist(bigrams)
            for bigram in vocabylary_data:
                if bigram in index.keys():
                    # Descriptiva
                    tf = vocabylary_data[bigram]
                    idf_per_word_and_document[bigram][docs] += 1
                    bow[docs, index[bigram]] = log10(tf + 1)

        # Discriminativa
        for bigram in index.keys():
            idf = sum(idf_per_word_and_document[bigram].values())
            idf = log10(n / idf)
            for docs, tweet in enumerate(data):
                bow[docs, index[bigram]] = bow[docs, index[bigram]] * idf
        return bow
