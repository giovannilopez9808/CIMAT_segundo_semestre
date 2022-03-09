from numpy import array, log, zeros, exp
from nltk import FreqDist, ngrams
from .functions import tokenize
from numpy.linalg import norm
from tabulate import tabulate
from random import choice


class probability_model_class:
    """
    Contenido para enmascarar palabras que no estan contenidas en un vocabulario y calculo de las probabilidades de unigramas, bigramas y trigramas
    """

    def __init__(self) -> None:
        pass

    def obtain_unigram_probabilities_Laplace(self, data: list) -> dict:
        """
        Calcula las probabilidades de cada unigrama en los datos dado
        --------------
        Inputs:
        + data -> lista de strings con cada tweet

        Output:
        + probability -> diccionario con las probabilidades de cada palabra en el vocabulario
        """
        # Tokens de cada oracion
        tokens = []
        for tweet in data:
            # Token de cada tweet
            tokens += tokenize(tweet)
        # Numero de palabras en el documento
        self.words_len = len(tokens)
        # Frecuencias de las palabras
        self.unigram_fdist = FreqDist(tokens)
        # Tamaño del vocabulario
        n_vocabulary = len(self.unigram_fdist)
        # Probabilidades de unigramas
        probability = {
            word: (count + 1.0) / (self.words_len + n_vocabulary)
            for word, count in self.unigram_fdist.items()
        }
        self.unigram_probability = probability.copy()
        return probability

    def obtain_ngrams_probabilities_Laplace(self, data: list, ngram: int) -> dict:
        """
        Calcula las probabilidades de cada ngrama en los datos dado
        --------------
        Inputs:
        + data -> lista de strings con cada tweet
        + ngram -> entero que indica el número de ngramas a calcular

        Output:
        + probability -> diccionario con las probabilidades de cada bigrama en el vocabulario
        """
        # Creacion de la lista de tokens
        tokens = []
        for tweet in data:
            # Tokens de cada tweet
            tokens += tokenize(tweet)
        self.tokens = tokens.copy()
        # ngramas en el documento
        n_grams = ngrams(tokens, n=ngram)
        # frecuencia de cada ngrama
        ngram_fdist = FreqDist(n_grams)
        # numero total de ngramas
        ngram_len = len(ngram_fdist)
        # inicializacion de la probabilidades
        probabilities = {}
        # Apartado para bigrama
        if ngram == 2:
            for (word_i, word_j), count in ngram_fdist.items():
                # Frecuencia de la palabra anterior en el bigrama
                count_word_i = self.unigram_fdist[word_i]
                # Probabilidad del bigrama
                probabilities[(word_j, word_i)] = (
                    count + 1.0) / (count_word_i + ngram_len)
            self.bigram_probability = probabilities.copy()
        # Apartado para trigrama
        if ngram == 3:
            # bigramas en el documento
            bigrams = ngrams(tokens, n=2)
            # frecuencias de cada bograma
            self.bigram_fdist = FreqDist(bigrams)
            for (word_i, word_j, word_k), count in ngram_fdist.items():
                # frecuencia del bigrama
                count_fs = self.bigram_fdist[(word_i, word_j)]
                # Suavizado con laplace
                probabilities[(word_k, word_i, word_j)] = (
                    count + 1.0) / (count_fs + ngram_len)
            self.trigram_probability = probabilities.copy()
        return probabilities

    def obtain_unigram_probability(self, word: str) -> float:
        """
        Obtiene la probabilidad que tiene un tweet siguiendo una probabilidad de unigramas
        --------------
        Input:
        + tweet -> string del tweet
        """
        if not word in self.unigram_probability:
            word = "<unk>"
        probability = self.unigram_probability[word]
        return probability

    def compute_ngram_probability(self,
                                  tweet: str,
                                  vocabulary: list,
                                  ngram: int = 2) -> float:
        """
        Calculo de la probabilidad de la oracion dada
        ----------
        Inputs:
        + tweet -> string tweet a calcular su probabilidad
        + vocabulary -> vocabulario de los datos de entrenamiento
        + ngram -> entero que indica la cantidad de ngramas a tomar

        -----------
        Output:
        probability -> probabilidad del tweet dado
        """
        # enmascaro oracion
        tweet = mask_unknow(tweet, vocabulary)
        # Concateno inicio y fin de oración
        s_init = "<s>"
        s_fin = "</s>"
        tweet = "{}{}{}".format(s_init, tweet, s_fin)
        tweet = tokenize(tweet)
        # Tamaño del vocabulario
        vocabulary_len = len(vocabulary)
        probability = 1
        # Apartado para bigrama
        if ngram == 2:
            # Frecuencia de los unigramas
            for i in range(len(tweet) - 1):
                word_i = tweet[i]
                word_j = tweet[i + 1]
                if (word_j, word_i) in self.bigram_probability:
                    probability *= self.bigram_probability[(word_j, word_i)]
                # Si la palabra no se encuentra se da una probabilidad
                else:
                    freq_word_i = self.unigram_fdist[word_i]
                    probability *= 1.0 / (freq_word_i + vocabulary_len)
        # Apartado para trigramas
        if ngram == 3:
            # Bigramas en la oracion
            bigrams = ngrams(self.tokens, n=2)
            # Freceucnia de bigramas
            self.bigrams_fdist = FreqDist(bigrams)
            # Tamaño e bigramas
            bigram_len = len(self.bigrams_fdist)
            for i in range(len(tweet) - 2):
                word_i = tweet[i]
                word_j = tweet[i + 1]
                word_k = tweet[i + 2]
                if (word_k, word_i, word_j) in self.trigram_probability:
                    probability *= self.trigram_probability[(
                        word_k, word_i, word_j)]
                # Si no se encuentra se da una probabilidad
                else:
                    count_fs = 0
                    if (word_i, word_j) in self.bigrams_fdist:
                        count_fs = self.bigrams_fdist[(word_i, word_j)]
                    conditional_prop = 1 / (count_fs + bigram_len)
                    probability *= conditional_prop
        return probability

    def obtain_ngram_probability(self, ngram: list) -> float:
        """
        Obtiene la probabilidad de un ngrama dado
        -----------
        Inputs:
        + ngram -> lista de ngramas

        -----------
        Output:
        + probability -> probabilidad del ngrama dado
        """
        probability = 1.0
        ngram_len = len(ngram)
        # Caso para bigrama
        if ngram_len == 2:
            for i in range(ngram_len - 1):
                word_i = ngram[i]
                word_j = ngram[i + 1]
                if (word_j, word_i) in self.bigram_probability:
                    probability *= self.bigram_probability[(word_j, word_i)]
                # Si la probabilidad no existe se da con al menos un conteo
                else:
                    freq_word_i = self.unigram_fdist[word_i]
                    probability *= 1.0 / (freq_word_i + self.words_len)
        # Caso para trigramas
        if ngram_len == 3:
            for i in range(ngram_len - 2):
                word_i = ngram[i]
                word_j = ngram[i + 1]
                word_k = ngram[i + 2]
                if (word_k, word_i, word_j) in self.trigram_probability:
                    probability *= self.trigram_probability[(
                        word_k, word_i, word_j)]
                # Si la probabilidad no existe se da con al menos un conteo
                else:
                    count_fs = 0.0
                    if (word_i, word_j) in self.bigram_fdist:
                        count_fs = self.bigram_fdist[(word_i, word_j)]
                    conditional_prop = 1.0 / (count_fs + self.words_len)
                    probability *= conditional_prop
        return probability


class language_model_class:

    def __init__(self, data_tr: list, data_test: list, data_val: list, vocabulary: dict):
        self.probability_model = probability_model_class()
        self.data_tr = data_tr
        self.data_test = data_test
        self.data_val = data_val
        self.vocabulary = vocabulary.keys()
        self.obtain_probabilities()

    def obtain_probabilities(self) -> None:
        self.unigram_probability = self.probability_model.obtain_unigram_probabilities_Laplace(
            self.data_tr)
        self.bigram_probability = self.probability_model.obtain_ngrams_probabilities_Laplace(
            self.data_tr,
            ngram=2,
        )
        self.tigram_probability = self.probability_model.obtain_ngrams_probabilities_Laplace(
            self.data_tr,
            ngram=3,
        )
        # Preparo modelo para evaluación
        tokens = []
        for tweet in self.data_tr:
            tokens += tokenize(tweet)
        # Total de palabras
        self.words_len = len(tokens)
        bigrams = ngrams(tokens, n=2)
        # Frecuencias bigramas
        self.bigram_fdist = FreqDist(bigrams)
        # Frecuencias unigramas
        self.unigram_fdist = FreqDist(tokens)

    def compute_perplexity(self, lambda_values: list, use_data_test: bool = True) -> float:
        tokens = []
        if use_data_test:
            data = self.data_test
        else:
            data = self.data_val
        for tweet in data:
            tokens += tokenize(tweet)
        trigrams = ngrams(tokens, n=3)
        perplexity = 0.0
        for (word_i, word_j, word_k) in trigrams:
            aux = 0.0
            aux += lambda_values[0] * self.probability_model.obtain_ngram_probability(
                (word_i, word_j, word_k),)
            aux += lambda_values[1] * self.probability_model.obtain_ngram_probability(
                (word_i, word_j),)
            aux += lambda_values[2] * self.probability_model.obtain_unigram_probability(
                word_i)
            perplexity += log(aux)
        perplexity = -perplexity / self.words_len
        return perplexity

    def tweet_probability(self, tweet: str, lambda_values: list, add_s_tokens: bool = True) -> float:
        """
        Calcula la probabilidad de un tweet por medio de la interpolacion
        ----------------
        Inputs:
        + tweet -> string del tweet a calcular su probabilidad
        + lambda_values -> lista de lambdas para los pesos de cada ngrama

        ----------------
        Outputs:
        + probability -> probabilidad del tweet dado
        """
        # Enmascaro palabras desconocidas
        if add_s_tokens:
            tweet = mask_unknow(tweet, self.vocabulary)
            tweet = "<s>{}</>".format(tweet)
        tokens = tokenize(tweet)
        trigrams = ngrams(tokens, n=3)
        probability = 1.0
        for (word_i, word_j, word_k) in trigrams:
            aux = 0
            # Compruebo valores de lambda
            if lambda_values[0] != 0:
                aux += lambda_values[0] * self.probability_model.obtain_ngram_probability(
                    (word_i, word_j, word_k),)
            if lambda_values[1] != 0:
                aux += lambda_values[1] * self.probability_model.obtain_ngram_probability(
                    (word_i, word_j),)
            if lambda_values[2] != 0:
                aux += lambda_values[2] * self.probability_model.obtain_unigram_probability(
                    word_i)
            probability *= aux
        return probability

    def apply_expectation_maximization(self, lambda_test: list = [], iterations: int = 5) -> array:
        results = []
        ngrams = 3
        if not len(lambda_test):
            lambda_test = [1/ngrams for i in range(ngrams)]
        perplexity = exp(self.compute_perplexity(lambda_test))
        results += [["Inicio",
                     lambda_test.copy(),
                     sum(lambda_test),
                     perplexity]]
        data_len = len(self.data_val)
        # Vectores de distribuciones q_m
        dist = zeros((data_len, ngrams), dtype=float)
        for iteration in range(iterations):
            # Ciclo sobre tokens de validación
            for i, tweet in enumerate(self.data_val):
                dist[i, 0] = self.tweet_probability(tweet,
                                                    [lambda_test[0], 0, 0],
                                                    add_s_tokens=False)
                dist[i, 1] = self.tweet_probability(tweet,
                                                    [0, lambda_test[1], 0],
                                                    add_s_tokens=False)
                dist[i, 2] = self.tweet_probability(tweet,
                                                    [0, 0, lambda_test[2]],
                                                    add_s_tokens=False)
                # Normalizo vector
                dist[i] = dist[i] / norm(dist[i])
            # Update lambdas
            for i in range(ngrams):
                lambda_test[i] = sum(dist[:, i]) / data_len
            perplexity = exp(self.compute_perplexity(lambda_test))
            results += [["Iteración {}".format(iteration+1),
                         lambda_test.copy(),
                         sum(lambda_test),
                         perplexity]]

        print(tabulate(results,
                       headers=["Iteracion", "lambdas", "Suma", "Perplexidad"]))
        return lambda_test


class tweetear_model:
    def __init__(self, language_model: language_model_class, lambdas: list):
        self.language_model = language_model
        self.lambdas = lambdas

    def autocomplete(self, init_text: list):
        # Creo todas las posibles oraciones
        tweets = []
        for word in self.language_model.vocabulary:
            value = "{} {}".format(init_text, word)
            tweets += [value]
        probabilities = []
        for i, tweet in enumerate(tweets):
            probabilities += [[self.language_model.tweet_probability(tweet,
                                                                     self.lambdas), i]]
        # Ordeno oraciones por probabilidad
        probabilities.sort(reverse=True)
        tweet = tweets[probabilities[0][1]]
        return tweet

    def write(self, init_text: list, max_words: int = 50) -> str:
        text = tokenize(init_text)
        result = text.copy()
        for i in range(max_words):
            tweet = self.autocomplete(" ".join(text))
            words = tokenize(tweet)
            word = words[-1]
            text.pop(0)
            text.append(word)
            result += [word]
            if word == '</s>':
                break
        result = " ".join(result)
        return result


def mask_unknow(tweet: str, vocabulary: list) -> str:
    """
        Enmascaramiento de una oración dado un vocabulario
        -----------
        Inputs:
        + tweet -> string con el tweet a enmascarar
        + vocabulary -> vocabulario de los datos de entrenamiento

        ------------
        Outputs:
        tweet_mask -> string con el tweet enmascarado
        """
    # Tokens del tweet dado
    tokens = tokenize(tweet)
    # Enmascaramiento de los tokens
    tweet_mask = [word if word in vocabulary else "<unk>"
                  for word in tokens]
    # Union de los tokens
    tweet_mask = " ".join(tweet_mask)
    return tweet_mask
