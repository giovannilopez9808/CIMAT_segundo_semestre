from cv2 import transpose
from numpy import sum, log10, log2, zeros, array, nonzero
from nltk.tokenize import TweetTokenizer as tokenizer
from sklearn.preprocessing import normalize
from numpy.random import choice, randint
from itertools import combinations
from numpy import matmul


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
                    t_ij = 1+log10(t_ij)
                    t_ij *= log10(vocabulary_len/sum_i)
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
                        value = log2(p_ij/(p_word*p_contex))
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
        log = log10(vocabulary_dim/vocabylary_document)
        for term in nonzeros_position:
            # Calculo de cada termino
            dor[term, i] = (1+log10(document[term])) * log
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
