from numpy import sum, log, zeros, array, nonzero, dot
from nltk.tokenize import TweetTokenizer as tokenizer
from sklearn.preprocessing import normalize
from numpy.random import choice, randint
from Modules.functions import join_path
from itertools import combinations
from tabulate import tabulate


def build_TCOR(data: list, vocabulary: dict, index_word: dict) -> array:
    """
    Método para crear una TCOR en base a unos datos dados. Este realiza un pesado de texto
    """
    vocabulary_len = len(vocabulary)
    tcor = zeros((vocabulary_len,
                  vocabulary_len),
                 dtype=float)
    # Conjunto de sets de los tokens por documento
    sets = [set(tokenizer().tokenize(doc)) for doc in data]
    # Palabras que no se encuentran en el vocabulario
    for subset in sets:
        # Copia del subset
        auxiliar = subset.copy()
        for word in auxiliar:
            # Si la palabra ya se encuentra en el set la elimina
            if word not in index_word:
                subset.remove(word)
    # Recorrido por todos los tweets
    for subset in sets:
        for word in subset:
            # Coocurrencia de palabra consigo mismas
            word_i = index_word[word]
            tcor[word_i, word_i] += 1.0
        # Comparacion si dos palabras se encuentra en el mismo documento
        for word_i, word_j in combinations(subset, 2):
            # Si se encuentran se cuentan
            if word_i in index_word and word_j in index_word:
                i = index_word[word_i]
                j = index_word[word_j]
                tcor[i, j] += 1.0
                tcor[j, i] += 1.0
    for i in range(vocabulary_len):
        # Suma de todos los valores mayores a cero de la columna
        sum_i = sum(tcor[i] > 0)
        # Recorrido por el vector
        for j in range(vocabulary_len):
            # Elemento ij de TCOR
            t_ij = tcor[i, j]
            # Si es mayor a cero
            if t_ij > 0:
                # Se calcula su peso
                t_ij = 1+log(t_ij)
                t_ij *= log(vocabulary_len/sum_i)
                tcor[i, j] = t_ij
    # Normalizacion
    tcor = normalize(tcor)
    return tcor


def build_DOR(bow: array) -> array:
    """
    Creacion de una DOR en base a una bolsa de palabras
    """
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
        log_value = log(vocabulary_dim/vocabylary_document)
        for term in nonzeros_position:
            # Calculo de cada termino
            dor[term, i] = (1+log(document[term])) * log_value
    dor = normalize(dor)
    return dor


def random_indexing(data: array, index_word: dict, size: int) -> array:
    """
    Creacion de random indexing a partir de una base de datos
    """
    vocabulary_len = len(index_word)
    # Representación random Indexing
    ri_matrix = zeros((vocabulary_len, size),
                      dtype=float)
    # Matriz de 0, -1, 1
    id_matrix = zeros((vocabulary_len, size), dtype=float)
    # Asigno aleatoriamente -1 y 1 al 20% de los datos
    nonzero_size = round(0.2 * size)
    for i in range(vocabulary_len):
        # Eleccion aleatoria de los -1,1
        values = choice([-1, 1], size=nonzero_size)
        # Eleccion aleatoria de las posicones
        positions = randint(size, size=nonzero_size)
        # Guardado de los valores aleatorios
        for j in range(nonzero_size):
            id_matrix[i, positions[j]] = values[j]
    # Suma de los contextos en cada espacio
    for doc in data:
        for word in tokenizer().tokenize(doc):
            if word in index_word:
                index = index_word[word]
                ri_matrix[index] += id_matrix[index]
    ri_matrix = normalize(ri_matrix)
    return ri_matrix


def tcor_to_BoW(data: array, vocabulary: list, index_word: dict, tcor: array) -> array:
    """
    Creacion de una BoW a partir de TCOR
    """
    # Tamaño de documetos
    document_len = len(data)
    # Tamaño de vocabulario
    vocabulary_len = len(vocabulary)
    # Obtengo conjunto de tokens por documento
    sets = [set(tokenizer().tokenize(doc)) for doc in data]
    # Eliminacion de las palabras que no se encuentran en el vocabulario
    for subset in sets:
        auxiliar = subset.copy()
        for word in auxiliar:
            if word not in index_word:
                subset.remove(word)
    # Inicialziacion de la BoW
    BoW = zeros((document_len, vocabulary_len),
                dtype=float)
    # Calculo de los pesos
    for i, subset in enumerate(sets):
        for word in subset:
            BoW[i] += tcor[index_word[word]]
        BoW[i] = BoW[i]/len(subset)
    return BoW


def obtain_cosine_similitud(data: array) -> list:
    """
    Obtiene la distancia de coseno de los vectores dado una matriz de datos
    """
    distances = []
    for i in range(data.shape[0]-1):
        pairs = range(i+1, data.shape[0])
        for j in pairs:
            distance = dot(data[i], data[j])
            distances.append([distance, i, j])
    distances.sort(reverse=True)
    return distances


def write_top_similitud_words(distances: array, word_index: dict, parameters: dict, show: bool = False):
    """
    Impresion de los resultsdos en la similitud de palabras
    """
    top_list = []
    i = 0
    while(i < parameters["max similitud words"]):
        data_i = distances[i]
        distance_i = data_i[0]
        index_i = data_i[1]
        index_j = data_i[2]
        if word_index[index_i] != word_index[index_j]:
            top_list += [[i+1,
                          distance_i,
                          word_index[index_i],
                          word_index[index_j]]]
            i += 1
    filename = join_path(parameters["path results"],
                         parameters["top words file"])
    table = tabulate(
        top_list,
        headers=[
            'Posición',
            'Angulo',
            'Palabra 1',
            'Palabra 2',
        ],
    )
    if show:
        print(table)
    else:
        file = open(filename, "w")
        file.write(table)
        file.close()


def write_top_similitud_documents(distances: array, data: array, parameters: dict, show: bool = False):
    """
    Escritura de la similitud de documentos
    """
    top_list = []
    for i in range(parameters["max similitud documents"]):
        data_i = distances[i]
        distance_i = data_i[0]
        index_i = data_i[1]
        index_j = data_i[2]
        top_list += [[i+1,
                     distance_i,
                     data[index_i],
                     data[index_j]]]
    filename = join_path(parameters["path results"],
                         parameters["top documents file"])
    table = tabulate(
        top_list,
        headers=[
            'Posición',
            'Angulo',
            'Documento 1',
            'Documento 2',
        ],
    )
    if show:
        print(table)
    else:
        file = open(filename, "w")
        file.write(table)
        file.close()
