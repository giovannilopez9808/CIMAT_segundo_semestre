from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame, read_csv, to_datetime
from .datasets import parameters_model
from .functions import join_path
from nltk.text import Text
from pandas import concat
from cmath import nan


class tripadvisor_model:
    def __init__(self, dataset: parameters_model) -> None:
        self.parameters = dataset.parameters
        self.age_range = {"Joven": [10, 50],
                          "Mayor": [51, 100]}

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
        data["new scale"] = data["Escala"].apply(self.new_scala_of_scores)
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

    def select_data_per_gender(self, gender: str) -> DataFrame:
        self.data_select = self.data[self.data["Género"] == gender]

    def select_data_per_nationality(self, nationality: str) -> DataFrame:
        self.data_select = self.data[self.data["Nacional ó Internacional"] == nationality]

    def select_data_per_age_range(self, age_range: str) -> DataFrame:
        age_range = self.age_range[age_range]
        self.data_select = self.data[self.data["Edad"] >= age_range[0]]
        self.data_select = self.data_select[self.data_select["Edad"]
                                            <= age_range[1]]

    def obtain_daily_counts_of_scores(self) -> DataFrame:
        results = {}
        result_basis = {0: 0,
                        1: 0,
                        2: 0}
        dates = sorted(list(set(self.data["Fecha"])))
        for date in dates:
            data_per_day = self.data[self.data["Fecha"] == date]
            data_counts = data_per_day["new scale"].value_counts()
            results[date] = result_basis.copy()
            for value in data_counts.index:
                results[date][value] = data_counts[value]
        results = DataFrame(results)
        self.daily_scores_counts = results.transpose()

    def obtain_yearly_stadistics_of_scores(self) -> DataFrame:
        self.obtain_daily_counts_of_scores()
        yearly_scores = self.daily_scores_counts.resample("Y").sum()
        for date in yearly_scores.index:
            year_sum = yearly_scores.loc[date].sum()
            if year_sum != 0:
                yearly_scores.loc[date] = yearly_scores.loc[date]/year_sum
            else:
                yearly_scores.loc[date] = nan
        yearly_mean_data = self.data.resample("Y", on="Fecha").mean()
        yearly_std_data = self.data.resample("Y", on="Fecha").std()
        self.yearly_data = concat([yearly_scores,
                                   yearly_mean_data["Escala"],
                                   yearly_std_data["Escala"]],
                                  axis=1)
        self.yearly_data.columns = [0, 1, 2, "Escala mean", "Escala std"]

    def obtain_monthly_stadistics_of_scores(self) -> DataFrame:
        self.obtain_daily_counts_of_scores()
        monthly_scores = self.daily_scores_counts.resample("MS").sum()
        for date in monthly_scores.index:
            monthly_sum = monthly_scores.loc[date].sum()
            if monthly_sum != 0:
                monthly_scores.loc[date] = monthly_scores.loc[date]/monthly_sum
            else:
                monthly_scores.loc[date] = nan
        monthly_mean_data = self.data.resample("MS", on="Fecha").mean()
        monthly_std_data = self.data.resample("MS", on="Fecha").std()
        self.monthly_data = concat([monthly_scores,
                                   monthly_mean_data["Escala"],
                                   monthly_std_data["Escala"]],
                                   axis=1)
        self.monthly_data.columns = [0, 1, 2, "Escala mean", "Escala std"]

    def obtain_only_negatives_scores(self):
        self.data_select = self.data[self.data["new scale"] == 0]
        print(len(self.data_select))
        if len(self.data_select) < 10:
            self.data_select = self.data[self.data["new scale"] != 2]

    def sort_by_date(self):
        self.data = self.data.sort_values(by="Fecha")

    def obtain_opinions_as_text(self, stopwords: list) -> Text:
        vectorizer = CountVectorizer(stop_words=stopwords)
        tokenizer = vectorizer.build_tokenizer()
        opinions = " ".join(self.data["Opinión"])
        opinions = tokenizer(opinions)
        opinions = Text(opinions)
        return opinions
