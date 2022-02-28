from pandas import DataFrame, read_csv, to_datetime
from .datasets import parameters_model
from .functions import join_path


class tripadvisor_model:
    def __init__(self, dataset: parameters_model) -> None:
        self.parameters = dataset.parameters

    def read_data(self,  name: str) -> DataFrame:
        """
        Lectura de los datos y formateo de la fecha dado el nombre del archivo
        """
        # Nombre con la ruta del archivo
        filename = join_path(self.parameters["path data"],
                             name)
        #  Lectura de los datos
        data = read_csv(filename)
        # Formato de los datos
        self.data = self.format_data(data)

    def format_data(self, data: DataFrame) -> DataFrame:
        """
        Formato de fecha a todo el dataframe
        """
        data = self.format_date(data)
        data = self.obtain_new_scala_of_scores(data)
        data = self.clean_text(data)
        return data

    def format_date(self, data: DataFrame) -> DataFrame:
        data["Fecha"] = data["Fecha"].astype(str).str.split("/")
        data["Fecha"] = data["Fecha"].apply(self.date_format)
        data["Fecha"] = to_datetime(data["Fecha"])
        return data

    def clean_text(self, data: DataFrame) -> DataFrame:
        columns = ["Título de la opinión", "Opinión"]
        for column in columns:
            data[column] = data[column].astype(str).str.replace('"', "")
            data[column] = data[column].astype(str).str.lower()
        return data

    def obtain_word_length_per_opinion(self) -> None:
        """
        Obtiene la cantidad de palabras por opinion
        """
        self.data["Word length"] = self.data["Opinión"].astype(str).str.split()
        self.data["Word length"] = self.data["Word length"].apply(len)

    def obtain_new_scala_of_scores(self, data: DataFrame) -> DataFrame:
        data["new scala"] = data["Escala"].apply(self.new_scala_of_scores)
        return data

    def new_scala_of_scores(self, score: int) -> int:
        if score in [4, 5]:
            return 2
        if score in [3]:
            return 1
        if score in [1, 2]:
            return 0

    def date_format(self, date: list) -> str:
        """
        Formate de fecha de listas [dia,mes,año] a año-mes-dia
        """
        day = date[0].zfill(2)
        month = date[1].zfill(2)
        year = date[2]
        date = "{}-{}-{}".format(year,
                                 month,
                                 day)
        return date
