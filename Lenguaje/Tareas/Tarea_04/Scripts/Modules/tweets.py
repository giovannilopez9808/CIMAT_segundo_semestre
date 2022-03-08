from .models import language_model_class, mask_unknow
from nltk.tokenize import TweetTokenizer as tokenizer
from sklearn.model_selection import train_test_split
from .vocabulary import vocabulary_class
from .dictionary import dictionary_class
from .functions import join_path
from tabulate import tabulate
from numpy import exp


class tweets_data:
    def __init__(self, parameters: dict) -> None:
        self.vocabulary_model = vocabulary_class()
        self.dictionary = dictionary_class()
        self.tokenize = tokenizer().tokenize
        self.parameters = parameters
        self.read()
        self.add_s_tokens()
        self.obtain_vocabulary(use_mask=True)
        self.obtain_word_index()
        self.obtain_index_word()
        self.data_tr_mask = self.mask_tweets(self.data_tr_mask)
        self.data_val_mask = self.mask_tweets(self.data_val_mask)
        self.obtain_vocabulary(use_mask=True)
        self.obtain_word_index()
        self.obtain_index_word()

    def get_texts_from_file(self, path_data: str, path_labels: str) -> tuple:
        """
        Obtiene una lista de oraciones a partir de un texto con sus respectivas etiquetas
        """
        # Inicilizacion de las listas
        text = []
        labels = []
        # Apertura de los archivos
        with open(path_data, "r") as f_data, open(path_labels, "r") as f_labels:
            # Recoleccion de las oraciones
            for tweet in f_data:
                text += [tweet]
            # Recoleccion de las etiquedas
            for label in f_labels:
                labels += [label]
        # Etiquedas a enteros
        labels = list(map(int, labels))
        return text, labels

    def read(self) -> None:
        """
        Lectura de los datos de entrenamiento y validacion
        """
        # Definicion de las rutas de cada archivo de datos y validacion
        path_data_tr = join_path(
            self.parameters["path data"],
            self.parameters["train"]["data"],
        )
        path_label_tr = join_path(
            self.parameters["path data"],
            self.parameters["train"]["labels"],
        )
        path_data_val = join_path(
            self.parameters["path data"],
            self.parameters["validation"]["data"],
        )
        path_label_val = join_path(
            self.parameters["path data"],
            self.parameters["validation"]["labels"],
        )
        # Lectura de las oraciones y etiquetas de los datos de entrenamiento
        self.data_tr, self.labels_tr = self.get_texts_from_file(
            path_data_tr,
            path_label_tr,
        )
        # Lectura de las oraciones y etiquetas de los datos de validación
        self.data_val, self.labels_val = self.get_texts_from_file(
            path_data_val,
            path_label_val,
        )

    def add_s_tokens(self) -> None:
        self.data_tr_mask = ["<s>{}</s>".format(tweet)
                             for tweet in self.data_tr]
        self.data_val_mask = ["<s>{}</s>".format(tweet)
                              for tweet in self.data_val]

    def obtain_vocabulary(self, use_mask: bool) -> None:
        if use_mask:
            data = self.data_tr_mask
        else:
            data = self.data_tr
        self.vocabulary = self.vocabulary_model.obtain(data,
                                                       self.parameters["max words"])

    def obtain_word_index(self) -> None:
        self.word_index = self.dictionary.build_word_index(self.vocabulary)

    def obtain_index_word(self) -> None:
        self.index_word = self.dictionary.obtain_index_word(self.word_index)

    def mask_tweets(self, tweets: list) -> None:
        tweets_mask = []
        for tweet in tweets:
            tweet_mask = mask_unknow(tweet,
                                     self.vocabulary.keys())
            tweets_mask += [tweet_mask]
        return tweets_mask

    def obtain_data_test(self) -> None:
        self.data_tr_mask, self.data_test_mask = train_test_split(self.data_tr_mask,
                                                                  train_size=0.89,
                                                                  test_size=0.11,
                                                                  random_state=12345)

    def obtain_perplexity(self, use_data_test: bool) -> None:
        print("Calculando perplejidad")
        self.obtain_data_test()
        language_model = language_model_class(self.data_tr_mask,
                                              self.data_test_mask,
                                              self.data_val_mask,
                                              self.vocabulary)
        results = []
        for lambda_i in self.parameters["lambda list"]:
            perplexity = language_model.compute_perplexity(lambda_i,
                                                           use_data_test=use_data_test)
            results += [[lambda_i, exp(perplexity)]]
        print(tabulate(results, headers=["Lambda", "Perplexity"]))
