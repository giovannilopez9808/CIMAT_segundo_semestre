from sklearn.feature_selection import chi2, SelectKBest
from numpy import array, concatenate, zeros, log
from sklearn.manifold import TSNE
from tabulate import tabulate
from nltk import FreqDist


def obtain_best_features(bow: array, labels: array) -> list:
    features = SelectKBest(chi2, k=1000)
    features.fit(bow, labels)
    best_features = features.get_support(indices=True)
    return best_features


def obtain_target_matrix(index: dict, data: array, best_features: array) -> array:
    invert_index = {}
    for word in index:
        invert_index[index[word]] = word
    target_words = [invert_index[word] for word in best_features]
    target_matrix = array([data[index[word]] for word in target_words])
    return target_words, target_matrix


def obatin_reduced_matrix(target_matrix: array) -> array:
    tsne_sk = TSNE(init="pca", perplexity=30, n_components=2)
    reduce_matrix = tsne_sk.fit_transform(target_matrix)
    return reduce_matrix


def join_path(path: str, filename: str) -> str:
    """
    Une la direccion de un archivo con su nombre
    """
    return "{}{}".format(path, filename)


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


def obtain_information_gain(data_tr: array, labels_tr: array, data_val: array, labels_val: array, index_word: dict) -> float:
    """
    Calcula la ganancia de infomacion de toda la coleccion de una base de datos
    """
    # Concadenacion de los datos de entrenamiento
    data = concatenate((data_tr, data_val))
    # Concadenacion de las clases
    labels = concatenate((labels_tr, labels_val))
    # Numero de documentos
    nt = data.shape[0]
    # Numero de documentos de cada clase
    np = FreqDist(labels)
    # Numero de documentos que contiene una palabra en cada clase
    nip = obtain_nip(data, labels, np, index_word)
    # Numero de terminos en la coleccion
    ni = obtain_ni(nip)
    IG = {}
    # Calculo de la funcion
    for word in index_word:
        value = 0
        n_i = ni[word]
        for type_class in np:
            n_p = np[type_class]
            n_ip = nip[word][type_class]
            value += n_p*log(n_p/nt)
            value += -n_ip*log(n_ip/n_i)
            value += -(n_p-n_ip)*log(n_p-n_ip+1)
            value += (n_p-n_ip)*log(nt-n_i)
        value = -value/nt
        IG[word] = value
    return IG


def obtain_nip(data: list, labels: list, np: FreqDist, index_word: dict) -> dict:
    """
    Obtiene el numero de documentos de cada termino por clase
    """
    nip = {}
    # Tamaño de las clases
    np_len = len(np)
    # Inicializacion del diccionario de las clases
    zeros_dict = dict(zip(np.keys(), zeros(np_len)))
    for word in index_word:
        # Inicializacion de cada termino
        nip[word] = zeros_dict.copy()
        for i, doc in enumerate(data):
            # Si la palabra esta en el documento
            if word in doc.lower():
                # Clase del documento
                type_class = labels[i]
                # Conteo
                nip[word][type_class] += 1
    return nip


def obtain_ni(nip: dict) -> dict:
    """
    Obtiene el numero de terminos en la coleccion
    """
    ni = {}
    for word in nip:
        ni[word] = sum(nip[word].values())
    return ni


def print_tuple_as_table(data: list, max: int) -> None:
    table = []
    for i in range(max):
        table += [[data[i][0],
                   data[i][1]]]
    print(tabulate(table,
                   headers=["Palabra",
                            "Gain information"]))
