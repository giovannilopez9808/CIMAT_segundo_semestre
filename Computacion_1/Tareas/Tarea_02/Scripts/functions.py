from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from nltk.tokenize import TweetTokenizer
from tabulate import tabulate
from sklearn import metrics
from sklearn import svm
import numpy as np
import nltk


def join_path(path: str, filename: str) -> str:
    """
    Une la direccion de un archivo con su nombre
    """
    return "{}{}".format(path, filename)


def get_texts_from_file(path_data: str, path_labels: str) -> tuple:
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


def sort_freqdist(fdist: nltk.FreqDist) -> list:
    """
    Ordena la lista de distribucion de frecuencias de palabras de mayor frecuencia a menor
    """
    aux = [(fdist[key], key) for key in fdist]
    aux.sort()
    aux.reverse()
    return aux


def split_data(data: list, max_words: int) -> list:
    """
    Realiza la separacion de elementos en una lista dado el numero de elementos que se quieren conservar
    """
    return data[:max_words]


def obtain_fdist(data: list, max_words: int) -> list:
    """
    Obtiene la lista de una distribucion de frecuencias de palabras ordenada de mayor a menor a partir de una lista de oraciones
    """
    # Inicializacion del Tokenizador
    tokenizer = TweetTokenizer()
    # Inicializacion de la lista que guardara los tokens
    corpus_palabras = []
    for tweet in data:
        # Creacion y guardado de los tokens
        corpus_palabras += tokenizer.tokenize(tweet)
    # Creacion de la distribucion de frecuencias
    fdist = nltk.FreqDist(corpus_palabras)
    fdist = sort_freqdist(fdist)
    fdist = split_data(fdist, max_words)
    return fdist


def obtain_fdist_with_bigrams(data: list, max_words: int) -> list:
    """
    Obtiene la lista de una distribucion de frecuencias de palabras ordenada de mayor a menor a partir de una lista de oraciones
    """
    # Inicializacion del Tokenizador
    tokenizer = TweetTokenizer()
    # Inicializacion de la lista que guardara los tokens
    corpus_bigrams = []
    for tweet in data:
        # Creacion y guardado de los tokens
        corpus_bigrams += nltk.bigrams(tokenizer.tokenize(tweet))
    # Creacion de la distribucion de frecuencias
    fdist = nltk.FreqDist(corpus_bigrams)
    fdist = sort_freqdist(fdist)
    fdist = split_data(fdist, max_words)
    return fdist


def create_dictonary_of_index(fdist: list) -> dict:
    """
    Crea un diccionario con la posición de mayor a menor frecuencia de cada palabra. La llave es la palabra a consultar
    """
    # Inicializacion del diccionario
    index = dict()
    # Inicializacion de la posicion
    i = 0
    for weight, word in fdist:
        index[word] = i
        i += 1
    return index


def build_binary_bow(data: list, fdist: list, index: dict) -> np.array:
    """
    Creacion de la BoW usando pesos binarios
    """
    tokenizer = TweetTokenizer()
    bow = np.zeros((len(data), len(fdist)), dtype=float)
    docs = 0
    for tweet in data:
        fdist_data = nltk.FreqDist(tokenizer.tokenize(tweet))
        for word in fdist_data:
            if word in index.keys():
                bow[docs, index[word]] = 1
        docs += 1
    return bow


def build_binary_bow_with_probabilities(data: list, fdist: list, index: dict,
                                        probability: dict) -> np.array:
    """
    Creacion de la BoW usando pesos binarios
    """
    tokenizer = TweetTokenizer()
    bow = np.zeros((len(data), len(fdist)), dtype=float)
    docs = 0
    for tweet in data:
        fdist_data = nltk.FreqDist(tokenizer.tokenize(tweet))
        for word in fdist_data:
            if word in index.keys():
                bow[docs, index[word]] = 1
                if word in probability:
                    bow[docs, index[word]] = probability[word]
        docs += 1
    return bow


def build_binary_bigram_bow(data: list, fdist: list, index: dict) -> np.array:
    """
    Creacion de la BoW usando pesos binarios
    """
    tokenizer = TweetTokenizer()
    bow = np.zeros((len(data), len(fdist)), dtype=float)
    docs = 0
    for tweet in data:
        bigrams = nltk.bigrams(tokenizer.tokenize(tweet))
        fdist_data = nltk.FreqDist(bigrams)
        for bigram in fdist_data:
            if bigram in index.keys():
                bow[docs, index[bigram]] = 1
        docs += 1
    return bow


def build_frecuency_bow(data: list, fdist: list, index: dict) -> np.array:
    """
    Creacion de la BoW usando pesos basado en frecuencias
    """
    tokenizer = TweetTokenizer()
    bow = np.zeros((len(data), len(fdist)), dtype=float)
    docs = 0
    for tweet in data:
        fdist_data = nltk.FreqDist(tokenizer.tokenize(tweet))
        for word in fdist_data:
            if word in index.keys():
                bow[docs, index[word]] = tweet.count(word)
        docs += 1
    return bow


def build_frecuency_bow_with_probabilities(data: list, fdist: list,
                                           index: dict,
                                           probability: dict) -> np.array:
    """
    Creacion de la BoW usando pesos basado en frecuencias
    """
    tokenizer = TweetTokenizer()
    bow = np.zeros((len(data), len(fdist)), dtype=float)
    docs = 0
    for tweet in data:
        fdist_data = nltk.FreqDist(tokenizer.tokenize(tweet))
        for word in fdist_data:
            if word in index.keys():
                bow[docs, index[word]] = tweet.count(word)
                if word in probability:
                    bow[docs, index[word]] *= probability[word]
        docs += 1
    return bow


def build_frecuency_bigram_bow(data: list, fdist: list,
                               index: dict) -> np.array:
    """
    Creacion de la BoW usando pesos basado en frecuencias
    """
    tokenizer = TweetTokenizer()
    bow = np.zeros((len(data), len(fdist)), dtype=float)
    docs = 0
    for tweet in data:
        bigrams = nltk.bigrams(tokenizer.tokenize(tweet))
        fdist_data = nltk.FreqDist(bigrams)
        for bigram in fdist_data:
            if bigram in index.keys():
                bow[docs, index[bigram]] = np.log(fdist_data[bigram] + 1)
        docs += 1
    return bow


def create_empty_dictionary_of_words_and_documents(words: dict,
                                                   data: list) -> dict:
    """
    Crea un diccionario el cual contendra de forma ordenada el indice de cada palabra y su numero de frecuencias en una coleccion
    """
    freq_word_per_document = dict()
    word_count = dict()
    for i, tweet in enumerate(data):
        word_count[i] = 0
    for word in words:
        freq_word_per_document[word] = word_count
    return freq_word_per_document


def build_tfidf_bow(data: list, fdist: list, index: dict) -> np.array:
    """
    Creacion de la BoW usando pesos basado en frecuencias
    """
    tokenizer = TweetTokenizer()
    # Inicilizacion del bow
    bow = np.zeros((len(data), len(fdist)), dtype=float)
    # Total de oraciones
    n = len(data)
    # Inicializacion del diccionario que contiene la repeticion de cada palabra
    idf_per_word_and_document = create_empty_dictionary_of_words_and_documents(
        index.keys(), data)
    for docs, tweet in enumerate(data):
        # Frecuencias
        fdist_data = nltk.FreqDist(tokenizer.tokenize(tweet))
        for word in fdist_data:
            if word in index.keys():
                # Descriptiva
                tf = tweet.count(word)
                idf_per_word_and_document[word][docs] += 1
                bow[docs, index[word]] = np.log(tf + 1)

    # Discriminativa
    for word in index.keys():
        idf = sum(idf_per_word_and_document[word].values())
        idf = np.log(n / idf)
        for docs, tweet in enumerate(data):
            bow[docs, index[word]] = bow[docs, index[word]] * idf
    return bow


def build_tfidf_bow_with_probabilities(data: list, fdist: list, index: dict,
                                       probability: dict) -> np.array:
    """
    Creacion de la BoW usando pesos basado en frecuencias
    """
    tokenizer = TweetTokenizer()
    # Inicilizacion del bow
    bow = np.zeros((len(data), len(fdist)), dtype=float)
    # Total de oraciones
    n = len(data)
    # Inicializacion del diccionario que contiene la repeticion de cada palabra
    idf_per_word_and_document = create_empty_dictionary_of_words_and_documents(
        index.keys(), data)
    for docs, tweet in enumerate(data):
        # Frecuencias
        fdist_data = nltk.FreqDist(tokenizer.tokenize(tweet))
        for word in fdist_data:
            if word in index.keys():
                # Descriptiva
                tf = tweet.count(word)
                idf_per_word_and_document[word][docs] += 1
                bow[docs, index[word]] = np.log(tf + 1)

    # Discriminativa
    for word in index.keys():
        idf = sum(idf_per_word_and_document[word].values())
        idf = np.log(n / idf)
        for docs, tweet in enumerate(data):
            if word in probability:
                bow[docs, index[word]] *= probability[word]
            bow[docs, index[word]] *= idf
    return bow


def build_tfidf_bigram_bow(data: list, fdist: list, index: dict) -> np.array:
    """
    Creacion de la BoW usando pesos basado en frecuencias
    """
    tokenizer = TweetTokenizer()
    # Inicilizacion del bow
    bow = np.zeros((len(data), len(fdist)), dtype=float)
    # Total de oraciones
    n = len(data)
    # Inicializacion del diccionario que contiene la repeticion de cada palabra
    idf_per_word_and_document = create_empty_dictionary_of_words_and_documents(
        index.keys(), data)
    for docs, tweet in enumerate(data):
        # Frecuencias
        bigrams = nltk.bigrams(tokenizer.tokenize(tweet))
        fdist_data = nltk.FreqDist(bigrams)
        for bigram in fdist_data:
            if bigram in index.keys():
                # Descriptiva
                tf = fdist_data[bigram]
                idf_per_word_and_document[bigram][docs] += 1
                bow[docs, index[bigram]] = np.log(tf + 1)

    # Discriminativa
    for bigram in index.keys():
        idf = sum(idf_per_word_and_document[bigram].values())
        idf = np.log(n / idf)
        for docs, tweet in enumerate(data):
            bow[docs, index[bigram]] = bow[docs, index[bigram]] * idf
    return bow


def create_model(bow_tr: np.array, labels_tr: np.array) -> GridSearchCV:
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


def evaluate_model(bow_val: np.array, labels_val: np.array, grid: GridSearchCV,
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
    print(confusion_matrix(
        labels_val,
        y_pred,
    ))
    print(metrics.classification_report(
        labels_val,
        y_pred,
    ))
    return [name, precision, recall, fscore]


def normalize(bow: np.array) -> np.array:
    """
    Normalizacion de la BoW de dos dimensiones
    """
    # Copia de la BoW
    bow_norm = bow.copy()
    for i in range(bow.shape[0]):
        # Inicializacion de la norma
        norm = 0
        # Calculo de la norma
        norm += sum([value**2 for value in bow[i]])
        norm = np.sqrt(norm)
        # Estandarizacion de la norma
        bow_norm[i] = np.array([value / norm for value in bow[i]])
    return bow_norm


def build_BoE_from_EmoLex(filename: str) -> dict:
    """
    Creacion de una bolsa de emociones a partir de la base de datos de EmoLex
    """
    with open(filename, "r", encoding='utf-8') as file:
        # Inicializacion de los diccionarios
        words_dict = dict()
        scores = dict()
        # Salto del header
        for i in range(1):
            next(file)
        for line in file:
            # Lectura de la informacion
            data = line.split('\t')
            if data[1] != 'NO TRANSLATION':
                # Obtencion del score
                score = float(data[3])
                word = data[1].lower()
                if not word in words_dict:
                    words_dict[word] = data[2]
                    scores[word] = score
                elif score > scores[word]:
                    words_dict[word] = data[2]
                    scores[word] = score
    return words_dict


def build_BoE_from_SEL(filename: str) -> tuple:
    """
    Creacion de una bolsa de emociones a partir de la base de datos de SEL
    """
    # Apertura del archivo
    with open(filename, "r", encoding='latin-1') as file:
        # Inicializacion de los diccionarios
        words_emotions = dict()
        scores = dict()
        # Salto del header
        for i in range(1):
            next(file)
        # Lectura del archivo
        for line in file:
            # Split de los datos
            data = line.split('\t')
            # Score
            score = float(data[1])
            # Palabra en minusculas
            word = data[0].lower()
            # Si no se ha guardado se guarda
            if not word in words_emotions:
                words_emotions[word] = data[2].replace("\n", "")
                scores[word] = score
            # Si ya existe se comprueba que sea el que contiene mayor score
            elif score > scores[word]:
                words_emotions[word] = data[2].replace("\n", "")
                scores[word] = score
    return words_emotions, scores


def mask_emotion(tokens: list, word_emotions: dict) -> list:
    """
    Enmascara un tweet a partir de las BoE dadas
    """
    token_copy = tokens.copy()
    for i, word in enumerate(tokens):
        if word in word_emotions:
            token_copy[i] = word_emotions[word]
    return token_copy


def obtain_corpus_emotions(document: list, word_emotions: dict) -> list:
    """
    Obtiene todo un corpus de emociones enmascarando cada tweet con la bolsa de emociones dada
    """
    tokenizer = TweetTokenizer()
    # Copia del corpus
    document_copy = document.copy()
    for i, tweet in enumerate(document):
        tweet = tokenizer.tokenize(tweet)
        emotions = mask_emotion(
            tweet,
            word_emotions,
        )
        document_copy[i] = " ".join(emotions)
    return document_copy


def obtain_parameters() -> dict:
    """
    Obtiene las rutas y nombres de los archivos que seran usados
    """
    parameters = {
        # Ruta de los archivos
        "path data": "../Data/",
        # Archivos de entrenamiento
        "train": {
            "data": "mex_train.txt",
            "labels": "mex_train_labels.txt"
        },
        # Archivos de validacion
        "validation": {
            "data": "mex_val.txt",
            "labels": "mex_val_labels.txt"
        },
        # Archivo de EmoLex
        "EmoLex": "emolex.txt",
        # Archivo de SEL
        "SEL": "SEL.txt",
        "max words": 5000,
        "max bigrams": 1000,
    }
    return parameters


def load_data(parameters: dict) -> tuple:
    """
    Lectura de los datos de entrenamiento y validacion
    """
    # Definicion de las rutas de cada archivo de datos y validacion
    path_data_tr = join_path(
        parameters["path data"],
        parameters["train"]["data"],
    )
    path_label_tr = join_path(
        parameters["path data"],
        parameters["train"]["labels"],
    )
    path_data_val = join_path(
        parameters["path data"],
        parameters["validation"]["data"],
    )
    path_label_val = join_path(
        parameters["path data"],
        parameters["validation"]["labels"],
    )
    # Lectura de las oraciones y etiquetas de los datos de entrenamiento
    data_tr, labels_tr = get_texts_from_file(
        path_data_tr,
        path_label_tr,
    )
    # Lectura de las oraciones y etiquetas de los datos de validación
    data_val, labels_val = get_texts_from_file(
        path_data_val,
        path_label_val,
    )
    return data_tr, labels_tr, data_val, labels_val


def load_emolex_data(parameters: dict, data_tr: list, data_val: list) -> tuple:
    """
    Lectura del dataset de EmoLex
    """
    emolex_path = join_path(
        parameters["path data"],
        parameters["EmoLex"],
    )
    # Carga de la bolsa de emociones
    words_emotions = build_BoE_from_EmoLex(emolex_path)
    data_tr_emolex_emotions = obtain_corpus_emotions(
        data_tr,
        words_emotions,
    )
    data_val_emolex_emotions = obtain_corpus_emotions(
        data_val,
        words_emotions,
    )
    return data_tr_emolex_emotions, data_val_emolex_emotions


def load_sel_data(parameters: dict, data_tr: list, data_val: list) -> tuple:
    """
    Lectura del dataset de SEL
    """
    sel_path = join_path(
        parameters["path data"],
        parameters["SEL"],
    )
    # Carga de la bolsa de emociones de SEL
    words_emotions, scores = build_BoE_from_SEL(sel_path)
    data_tr_sel_emotions = obtain_corpus_emotions(
        data_tr,
        words_emotions,
    )
    data_val_sel_emotions = obtain_corpus_emotions(
        data_val,
        words_emotions,
    )
    return data_tr_sel_emotions, data_val_sel_emotions, scores


def print_results(results: list) -> None:
    """
    Impresion estandarizada de los resultados
    """
    print(
        tabulate(
            results,
            headers=[
                'Algoritmo',
                'Precision',
                'Recall',
                'F1 Score',
            ],
        ))
