class dictionary_model:
    """
    Métodos para crear diccionarios con diferentes informaciones
    """

    def __init__(self) -> None:
        pass

    def build_word_index(self, vocabylary: list) -> dict:
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

    def build_index_word(self, word_index: dict) -> dict:
        """
        Invierte los valores de un diccionario dado. Los values pasan a ser keys y viceversa
        """
        invert_index = {}
        for word in word_index:
            invert_index[word_index[word]] = word
        return invert_index

    def sort_dict(self, data: dict, reverse: bool = True) -> dict:
        """
        Ordena un diccionario
        """
        dict_sort = sorted(
            data.items(), key=lambda item: item[1], reverse=reverse)
        return dict_sort
