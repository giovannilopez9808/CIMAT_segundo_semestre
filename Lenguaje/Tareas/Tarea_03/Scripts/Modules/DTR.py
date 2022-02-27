from numpy import sum, log, log2, zeros, array, nonzero, dot
from nltk.tokenize import TweetTokenizer as tokenizer
from sklearn.preprocessing import normalize
from numpy.random import choice, randint
from itertools import combinations
from tabulate import tabulate

from Modules.functions import join_path


def build_TCOR(data: list, vocabulary: dict, index_word: dict, weight: str = 'short-text') -> array:
    vocabulary_len = len(vocabulary)
    tcor = zeros((vocabulary_len,
                  vocabulary_len),
                 dtype=float)
    # Conjunto de tokens por documento
    sets = [set(tokenizer().tokenize(doc)) for doc in data]
    # Palabras que no estan en el vocabulario
    for subset in sets:
        auxiliar = subset.copy()
        for word in auxiliar:
            if word not in index_word:
                subset.remove(word)
    for subset in sets:
        for word in subset:
            # Coocurrencia de palabra consigo mismas
            word_i = index_word[word]
            tcor[word_i, word_i] += 1.0
        for word_i, word_j in combinations(subset, 2):
            if word_i in index_word and word_j in index_word:
                i = index_word[word_i]
                j = index_word[word_j]
                tcor[i, j] += 1.0
                tcor[j, i] += 1.0

    if weight == 'short-text':
        for i in range(vocabulary_len):
            sum_i = sum(tcor[i] > 0)
            for j in range(vocabulary_len):
                t_ij = tcor[i, j]
                if t_ij > 0:
                    t_ij = 1+log(t_ij)
                    t_ij *= log(vocabulary_len/sum_i)
                    tcor[i, j] = t_ij
    if weight == 'PPMI':
        word_count = sum(tcor,
                         axis=1)
        context_count = sum(tcor,
                            axis=0)
        total = sum(word_count)
        for i in range(tcor.shape[0]):
            for j in range(tcor.shape[1]):
                t_ij = tcor[i, j]
                if t_ij > 0:
                    p_ij = t_ij / total
                    p_word = word_count[i] / total
                    p_contex = context_count[j] / total
                    if abs(p_word * p_contex) > 0:
                        value = log(p_ij/(p_word*p_contex))
                        tcor[i, j] = max(value, 0)
    tcor = normalize(tcor)
    return tcor


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
        log = log(vocabulary_dim/vocabylary_document)
        for term in nonzeros_position:
            # Calculo de cada termino
            dor[term, i] = (1+log(document[term])) * log
    dor = normalize(dor)
    return dor


def random_indexing_with_DOR(data: array, index_word: dict, size: int) -> array:
    vocabulary_len = len(index_word)
    # Representación random Indexing
    ri_matrix = zeros((vocabulary_len, size),
                      dtype=float)
    # Matriz de 0, -1, 1
    id_matrix = zeros((vocabulary_len, size), dtype=float)
    # Asigno aleatoriamente -1 y 1 al 20% de los datos
    nonzero_size = round(0.2 * size)
    for i in range(vocabulary_len):
        values = choice([-1, 1], size=nonzero_size)
        positions = randint(size, size=nonzero_size)
        for j in range(nonzero_size):
            id_matrix[i, positions[j]] = values[j]
    for doc in data:
        for word in tokenizer().tokenize(doc):
            if word in index_word:
                index = index_word[word]
                ri_matrix[index] += id_matrix[index]
    ri_matrix = normalize(ri_matrix)
    return ri_matrix


def tcor_to_BoW(data: array, vocabulary: list, index_word: dict, tcor: array) -> array:
    document_len = len(data)
    vocabulary_len = len(vocabulary)
    # Obtengo conjunto de tokens por documento
    sets = [set(tokenizer().tokenize(doc)) for doc in data]
    # Quito palabras que no están en vocabulario
    for subset in sets:
        auxiliar = subset.copy()
        for word in auxiliar:
            if word not in index_word:
                subset.remove(word)
    BoW = zeros((document_len, vocabulary_len),
                dtype=float)
    i = 0
    for subset in sets:
        n = 0
        for word in subset:
            BoW[i] += tcor[index_word[word]]
            n += 1
        BoW[i] = BoW[i]/n
        i += 1
    return BoW


def obtain_cosine_similitud(data: array) -> list:
    distances = []
    for i in range(data.shape[0]-1):
        pairs = range(i+1, data.shape[0])
        for j in pairs:
            distance = dot(data[i], data[j])
            distances.append([distance, i, j])
    distances.sort(reverse=True)
    return distances


def write_top_similitud_words(distances: array, word_index: dict, parameters: dict):
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
    file = open(filename, "w")
    file.write(
        tabulate(
            top_list,
            headers=[
                'Posición',
                'Angulo',
                'Palabra 1',
                'Palabra 2',
            ],
        ))
    file.close()


def write_top_similitud_documents(distances: array, data: array, parameters: dict):
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
    file = open(filename, "w")
    file.write(tabulate(
        top_list,
        headers=[
            'Posición',
            'Angulo',
            'Palabra 1',
            'Palabra 2',
        ],
    ))
    file.close()
