from sklearn.metrics import precision_recall_fscore_support
from nltk.tokenize import TweetTokenizer as tokenizer
from sklearn.model_selection import GridSearchCV
from nltk import FreqDist, ngrams
from numpy import array, log10
from sklearn import svm


class SVM_model:
    def __init__(self) -> None:
        pass

    def create_model(self, bow_tr: array, labels_tr: array) -> GridSearchCV:
        """
        Creacion del modelo para realizar el aprendizaje
        """
        parameters_model = {"C": [0.05, 0.12, 0.25, 0.5, 1, 2, 4]}
        svr = svm.LinearSVC(class_weight="balanced", max_iter=1200000)
        grid = GridSearchCV(estimator=svr,
                            param_grid=parameters_model,
                            n_jobs=8,
                            scoring="f1_macro",
                            cv=5)
        grid.fit(bow_tr, labels_tr)
        return grid

    def evaluate_model(self, bow_val: array, labels_val: array, grid: GridSearchCV,
                       name: str) -> list:
        """
        Resultados del modelo con el dataset de validacion
        """
        y_pred = grid.predict(bow_val)
        precision, recall, fscore, _ = precision_recall_fscore_support(
            labels_val,
            y_pred,
            average="macro",
            pos_label=1,
        )
        return [name, precision, recall, fscore]


class language_model_class:
    def __init__(self) -> None:
        self.tokenize = tokenizer().tokenize

    def mask_unknow(self, tweet: str, vocabulary: list) -> str:
        tokens = self.tokenize(tweet)
        tweet_mask = [word if word in vocabulary else "<unk>"
                      for word in tokens]
        tweet_mask = " ".join(tweet_mask)
        return tweet_mask

    def obtain_unigram_probabilities_Laplace(self, data: list):
        # Concateno oraciones con filtrado <unk>
        tokens = []
        for tweet in data:
            tokens += self.tokenize(tweet)
        # Total de palabras
        n_tweets = len(tokens)
        # Creo diccionario de frecuencias actualizadas
        unigram_fdist = FreqDist(tokens)
        n_vocabulary = len(unigram_fdist)
        # Creo diccionario de probabilidades con suavisado de Laplace
        probability = {word: (count + 1.0) / (n_tweets + n_vocabulary)
                       for word, count in unigram_fdist.items()}
        return probability

    def obtain_ngrams_probabilities_Laplace(self, data: list, vocabulary: list, ngram: int) -> dict:
        vocabulary_len = len(vocabulary)
        tokens = []
        for tweet in data:
            tokens += self.tokenize(tweet)
        # Creo diccionario con caracteres <s> y </s>
        n_grams = ngrams(tokens, n=ngram)
        ngram_fdist = FreqDist(n_grams)
        probabilities = {}
        # Calculo secuencias solo si aparecen
        if ngram == 2:
            freq_tokens = FreqDist(tokens)
            for (first, second), count in ngram_fdist.items():
                count_first = freq_tokens[first]
                probabilities[(second, first)] = (count + 1.0) / \
                    (count_first + vocabulary_len)
        if ngram == 3:
            bigrams = ngrams(tokens, n=ngram-1)
            bigram_fdist = FreqDist(bigrams)
            for (first, second, third), count in ngram_fdist.items():
                count_fs = bigram_fdist[(first, second)]
                # Suavizado con laplace
                probabilities[(third, first, second)] = (
                    count + 1.0) / (count_fs + vocabulary_len)
        return probabilities

    def compute_unigram_probability(self, tweet: str, unigram_probability: dict):
        probability = 1
        tweet = tweet.split()
        for word in tweet:
            if not word in unigram_probability:
                word = "<unk>"
            probability *= unigram_probability[word]
        return probability

    def compute_ngram_probability(self, tweet: str, ngram_probability: dict, data_masked: list, vocabulary: list, ngram: int = 2) -> float:
        # enmascaro oracion
        tweet = self.mask_unknow(tweet, vocabulary)
        # Concateno inicio y fin de oración
        s_init = "<s>"
        s_fin = "</s>"
        tweet = "{}{}{}".format(s_init,
                                tweet,
                                s_fin)
        tweet = self.tokenize(tweet)
        # Concateno oraciones con filtrado <unk>
        tokens = []
        for tweet_masked in data_masked:
            # text = " ".join(tweet_masked)
            tokens += self.tokenize(tweet_masked)
        # Creo diccionario de frecuencias actualizadas
        vocabulary_len = len(vocabulary)
        probability = 1
        if ngram == 2:
            unigram_fdist = FreqDist(tokens)
            for i in range(len(tweet) - 1):
                word_i = tweet[i]
                word_j = tweet[i+1]
                if (word_j, word_i) in ngram_probability:
                    probability *= ngram_probability[(word_j, word_i)]
                # Calculo probabilidad solo si es necesario
                else:
                    freq_word_i = unigram_fdist[word_i]
                    freq_word_i = 1.0 / (freq_word_i + vocabulary_len)
                    probability *= freq_word_i
        if ngram == 3:
            bigrams = ngrams(tokens, n=ngram - 1)
            bigrams_fdist = FreqDist(bigrams)
            for i in range(len(tweet) - 2):
                word_i = tweet[i]
                word_j = tweet[i+1]
                word_k = tweet[i+2]
                if (word_k, word_i, word_j) in ngram_probability:
                    probability *= ngram_probability[(word_k, word_i, word_j)]
                # Calculo probabilidad solo si es necesario
                else:
                    count_fs = 0
                    if (word_i, word_j) in bigrams_fdist:
                        count_fs = bigrams_fdist[(word_i, word_j)]
                    conditional_prop = 1 / (count_fs + vocabulary_len)
                    probability *= conditional_prop
        return probability

    def obtain_ngram_probability(self, ngram: list, ngram_probability: dict, unigram_fdist, words_len: int):
        probability = 1.0
        ngram_len = len(ngram)
        if ngram_len == 2:
            for i in range(ngram_len - 1):
                first = ngram[i]
                second = ngram[i+1]
                if (second, first) in ngram_probability:
                    probability *= ngram_probability[(second, first)]
                # Calculo probabilidad solo si es necesario
                else:
                    freq_first = unigram_fdist[first]
                    freq_first = 1.0 / (freq_first + words_len)
                    probability *= freq_first
        if ngram_len == 3:
            for i in range(ngram_len - 2):
                first = ngram[i]
                second = ngram[i+1]
                third = ngram[i+2]
                if (third, first, second) in ngram_probability:
                    probability *= ngram_probability[(third, first, second)]
                # Calculo probabilidad solo si es necesario
                else:
                    count_fs = 0.0
                    if (first, second) in unigram_fdist:
                        count_fs = unigram_fdist[(first, second)]
                    conditional_prop = 1.0 / (count_fs + words_len)
                    probability *= conditional_prop
        return probability


class perplexity_model_class:
    def __init__(self, data_tr: list, data_test: list, vocabulary: dict):
        self.languaje_model = language_model_class()
        self.tokenize = tokenizer().tokenize
        self.data_tr = data_tr
        self.data_test = data_test
        self.vocabulary = vocabulary.keys()
        self.obtain_probabilities()

    def obtain_probabilities(self) -> None:
        self.unigram_probability = self.languaje_model.obtain_unigram_probabilities_Laplace(
            self.data_tr)
        self.bigram_probability = self.languaje_model.obtain_ngrams_probabilities_Laplace(
            self.data_tr,
            self.vocabulary,
            ngram=2)
        self.tigram_probability = self.languaje_model.obtain_ngrams_probabilities_Laplace(
            self.data_tr,
            self.vocabulary,
            ngram=3)
        # Preparo modelo para evaluación
        tokens = []
        for tweet in self.data_tr:
            tokens += self.tokenize(tweet)
        # Total de palabras
        self.words_len = len(tokens)
        # Frecuencias bigramas
        bigrams = ngrams(tokens, n=2)
        self.bigram_fdist = FreqDist(bigrams)
        # Frecuencias unigramas
        self.fdist_unigramas = FreqDist(tokens)

    def compute_perplexity(self, lambda_values: list) -> float:
        # Agrego padding solo del lado derecho
        tokens = []
        for tweet in self.data_test:
            tokens += self.tokenize(tweet)
        trigrams = ngrams(tokens, n=3)
        perplexity = 0.0
        for (first, second, third) in trigrams:
            tmp = 0.0
            tmp += lambda_values[0] * self.languaje_model.obtain_ngram_probability(
                (first, second, third),
                self.tigram_probability,
                self.bigram_fdist,
                words_len=self.words_len)
            tmp += lambda_values[1] * self.languaje_model.obtain_ngram_probability(
                (first, second),
                self.bigram_probability,
                self.fdist_unigramas,
                words_len=self.words_len)
            tmp += lambda_values[2] * self.languaje_model.compute_unigram_probability(
                first,
                self.unigram_probability)
            perplexity += log10(tmp)
        perplexity = - 1.0 / self.words_len * perplexity
        return perplexity

    def sentence_probability(self, sentence: str, lambda_values) -> float:
        # Enmascaro palabras desconocidas
        sentence = self.languaje_model.mask_unknow(sentence,
                                                   self.vocabulary)
        sentence = "<s>{}</s>".format(sentence)
        tokens = self.tokenize(sentence)
        trigrams = ngrams(tokens, n=3)
        probability = 1.0
        for (first, second, third) in trigrams:
            tmp = 0.0
            # Compruebo valores de lambda
            if lambda_values[0] != 0.0:
                tmp += lambda_values[0] * self.languaje_model.obtain_ngram_probability(
                    (first, second, third),
                    self.tigram_probability,
                    self.bigram_fdist,
                    words_len=self.words_len,
                    ngram=3)
            if lambda_values[1] != 0.0:
                tmp *= lambda_values[1] * self.languaje_model.obtain_ngram_probability(
                    (first, second),
                    self.bigram_probability,
                    self.fdist_unigramas,
                    words_len=self.words_len,
                    ngram=2)
            if lambda_values[2] != 0.0:
                tmp += lambda_values[2] * self.languaje_model.compute_unigram_probability(
                    first,
                    self.unigram_probability)
            probability *= tmp
        return probability
