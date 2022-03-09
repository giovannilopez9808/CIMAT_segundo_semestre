from sklearn.model_selection import train_test_split
from .functions import join_path, ls, tokenize
from .vocabulary import vocabulary_class
from .dictionary import dictionary_class
from .tweets import tweets_data
from unidecode import unidecode


class AMLO_conferences_model(tweets_data):
    def __init__(self, parameters: dict) -> None:
        self.vocabulary_model = vocabulary_class()
        self.dictionary = dictionary_class()
        self.parameters = parameters
        self.read()
        self.obtain_data_val()
        self.initialize()

    def read(self) -> list:
        """
        Realiza la lectura de todas las conferencias y las reune en un solo string
        Input:
            String: path -> Direccion donde se encuentran todos los archivos

        Output:
            String: Texto plano
        """
        files = ls(self.parameters["path conferences"])
        self.data = []
        for file in files:
            # Direccion y nombre del archivo
            filename = join_path(self.parameters["path conferences"],
                                 file)
            # Apertura del archivo
            file_data = open(filename, "r", encoding="utf-8")
            file_text = file_data.read()
            file_text = file_text.lower()
            file_text = unidecode(file_text)
            file_text = tokenize(file_text)
            file_text = " ".join(file_text)
            # Concadenacion del texto
            self.data += [file_text]
            # Cierre del texto
            file_data.close()

    def obtain_data_val(self) -> None:
        self.data_tr, self.data_val = train_test_split(self.data,
                                                       train_size=0.9,
                                                       test_size=0.1,
                                                       random_state=12345)
