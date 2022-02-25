from nltk.tokenize import TweetTokenizer as tokenizer
from functions import sort_freqdist, split_data
from nltk import bigrams as bigrams_nltk
from nltk import FreqDist


class vocabularies:
    def __init__(self) -> None:
        pass

    def obtain(self, data: list, max_words: int) -> list:
        """
        Obtiene la lista de una distribucion de frecuencias de palabras ordenada de mayor a menor a partir de una lista de oraciones
        """
        # Inicializacion de la lista que guardara los tokens
        corpus_palabras = []
        for tweet in data:
            # Creacion y guardado de los tokens
            corpus_palabras += tokenizer().tokenize(tweet)
        # Creacion de la distribucion de frecuencias
        vocabylary = FreqDist(corpus_palabras)
        vocabylary = sort_freqdist(vocabylary)
        vocabylary = split_data(vocabylary, max_words)
        return vocabylary

    def obtain_with_bigrams(self, data: list, max_bigrams: int) -> list:
        """
        Obtiene la lista de una distribucion de frecuencias de palabras ordenada de mayor a menor a partir de una lista de oraciones
        """
        # Inicializacion de la lista que guardara los tokens
        corpus_bigrams = []
        for tweet in data:
            # Creacion y guardado de los tokens
            corpus_bigrams += bigrams_nltk(tokenizer().tokenize(tweet))
        # Creacion de la distribucion de frecuencias
        vocabulary = FreqDist(corpus_bigrams)
        vocabulary = sort_freqdist(vocabulary)
        vocabulary = split_data(vocabulary, max_bigrams)
        return vocabulary
