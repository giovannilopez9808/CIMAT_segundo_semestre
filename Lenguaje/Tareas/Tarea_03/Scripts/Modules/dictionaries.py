class dictionaries:
    def __init__(self) -> None:
        pass

    def build_of_index(self, vocabylary: list) -> dict:
        """
        Crea un diccionario con la posición de mayor a menor frecuencia de cada palabra. La llave es la palabra a consultar
        """
        # Inicializacion del diccionario
        index = dict()
        # Inicializacion de la posicion
        i = 0
        for weight, word in vocabylary:
            index[word] = i
            i += 1
        return index

    def build_with_words_and_documents(self, words: dict,
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

    def obtain_invert_dictionary(self, index_word: dict) -> dict:
        invert_index = {}
        for word in index_word:
            invert_index[index_word[word]] = word
        return invert_index
