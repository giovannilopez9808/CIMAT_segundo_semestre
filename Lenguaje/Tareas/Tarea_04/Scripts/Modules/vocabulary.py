from nltk.tokenize import TweetTokenizer as tokenizer
from .dictionary import dictionary_class
from nltk import bigrams as bigrams_nltk
from nltk import FreqDist


class vocabulary_class:
    def __init__(self) -> None:
        self.tokenize = tokenizer().tokenize
        pass

    def obtain(self, data: list, max_words: int) -> dict:
        """
        Obtiene la lista de una distribucion de frecuencias de palabras ordenada de mayor a menor a partir de una lista de oraciones
        """
        # Inicializacion de la lista que guardara los tokens
        corpus = []
        for tweet in data:
            # Creacion y guardado de los tokens
            corpus += self.tokenize(tweet)
        # Creacion de la distribucion de frecuencias
        vocabulary = FreqDist(corpus)
        vocabulary = self.sort_freqdist(vocabulary)
        # print(vocabulary)
        vocabulary = self.split_data(vocabulary, max_words)
        return vocabulary

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
        vocabulary = self.sort_freqdist(vocabulary)
        vocabulary = self.split_data(vocabulary, max_bigrams)
        return vocabulary

    def sort_freqdist(self, vocabulary: FreqDist) -> list:
        """
        Ordena la lista de distribucion de frecuencias de palabras de mayor frecuencia a menor
        """
        aux = {}
        for word in vocabulary:
            aux[word] = vocabulary[word]
        aux = dictionary_class().sort_dict(aux)
        return aux

    def split_data(self, data: dict, max_words: int) -> list:
        """
        Realiza la separacion de elementos en una lista dado el numero de elementos que se quieren conservar
        """
        aux = {}
        for i, word in enumerate(data):
            if i >= max_words:
                break
            aux[word] = data[word]
        return aux
