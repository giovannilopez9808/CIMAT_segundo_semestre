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
        data["Fecha"] = data["Fecha"].astype(str).str.split("/")
        data["Fecha"] = data["Fecha"].apply(self.format_date)
        data["Fecha"] = to_datetime(data["Fecha"])
        return data

    def format_date(self, date: list) -> str:
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

    def obtain_word_length_per_opinion(self):
        """
        Obtiene la cantidad de palabras por opinion
        """
        self.data["Word length"] = self.data["Opinión"].astype(str).str.split()
        self.data["Word length"] = self.data["Word length"].apply(len)
