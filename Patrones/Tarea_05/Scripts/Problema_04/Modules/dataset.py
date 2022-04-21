from pandas import DataFrame, read_csv
from os.path import join


class dataset_model:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.read()

    def read(self) -> DataFrame:
        filename = join(self.params["path data"],
                        self.params["file data"])
        self.data = read_csv(filename)
        self.data = self.shuffle_data(self.data)

    def shuffle_data(self, data: DataFrame) -> DataFrame:
        data_shuffle = data.sample(frac=1,
                                   random_state=42)
        data_shuffle = data_shuffle.reset_index(drop=True)
        return data_shuffle

    def select_cash_type(self, type_name: str) -> DataFrame:
        if type_name == "True":
            self.select_true_cash()
        if type_name == "False":
            self.select_false_cash()

    def select_true_cash(self) -> DataFrame:
        self.data_select = self.data[self.data["label"] == 1]

    def select_false_cash(self) -> DataFrame:
        self.data_select = self.data[self.data["label"] == 0]
