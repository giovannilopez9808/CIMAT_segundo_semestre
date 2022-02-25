from sklearn.feature_selection import chi2, SelectKBest
from nltk.tokenize import TweetTokenizer as tokenizer
from numpy import array, sqrt, zeros, nonzero, log10
from sklearn.manifold import TSNE
from sklearn import preprocessing
from nltk import FreqDist
from auxiliar import *
from tsne import tsne
from bows import *


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


def sort_freqdist(vocabylary: FreqDist) -> list:
    """
    Ordena la lista de distribucion de frecuencias de palabras de mayor frecuencia a menor
    """
    aux = [(vocabylary[key], key) for key in vocabylary]
    aux.sort()
    aux.reverse()
    return aux


def split_data(data: list, max_words: int) -> list:
    """
    Realiza la separacion de elementos en una lista dado el numero de elementos que se quieren conservar
    """
    return data[:max_words]


def normalize(bow: array) -> array:
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
        norm = sqrt(norm)
        # Estandarizacion de la norma
        bow_norm[i] = array([value / norm for value in bow[i]])
    return bow_norm


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
    # Copia del corpus
    document_copy = document.copy()
    for i, tweet in enumerate(document):
        tweet = tokenizer().tokenize(tweet)
        emotions = mask_emotion(
            tweet,
            word_emotions,
        )
        document_copy[i] = " ".join(emotions)
    return document_copy


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


def build_DOR(bow: array) -> array:
    bow_subset = bow.copy()
    # Vocabulario de la coleccion
    vocabulary_dim = bow_subset.shape[1]
    dor = zeros((vocabulary_dim,
                 bow_subset.shape[0]),
                dtype=float)
    for i, document in enumerate(bow_subset):
        # No zeros
        nonzeros_position = nonzero(document)[0]
        # Vocabulario en el documento
        vocabylary_document = len(nonzeros_position)
        # Logaritmo del numero de vocabularios
        log = log10(vocabulary_dim/vocabylary_document)
        for term in nonzeros_position:
            # Calculo de cada termino
            dor[term, i] = (1+log10(document[term])) * log
    dor = preprocessing.normalize(dor)
    return dor


def obtain_best_features(bow: array, labels: array) -> list:
    features = SelectKBest(chi2, k=1000)
    features.fit(bow, labels)
    best_features = features.get_support(indices=True)
    return best_features


def obtain_target_matrix(index: dict, dor: array, best_features: array) -> array:
    invert_index = {}
    for word in index:
        invert_index[index[word]] = word
    target_words = [invert_index[word] for word in best_features]
    target_matrix = array([dor[index[word]] for word in target_words])
    return target_matrix


def obatin_reduce_matrix(target_matrix: array) -> array:
    reduce_matrix = tsne(target_matrix, 2)
    return reduce_matrix


def obatin_reduce_matrix_sklearn(target_matrix: array) -> array:
    tsne_sk = TSNE(init="pca", perplexity=30, learning_rate=30)
    reduce_matrix = tsne_sk.fit_transform(target_matrix)
    return reduce_matrix
