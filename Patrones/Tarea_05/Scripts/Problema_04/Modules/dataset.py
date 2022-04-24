from pandas import DataFrame, read_csv
from os.path import join
from numpy import array


class dataset_model:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.read()

    def read(self) -> DataFrame:
        filename = join(self.params["path data"],
                        self.params["file data"])
        self.data = read_csv(filename)

    def select_data(self, names: list) -> array:
        subset = self.data[names].to_numpy()
        return subset

    def get_labels(self) -> array:
        labels = self.data["label"].to_numpy()
        return labels

    def select_cash_type(self, type_name: str) -> DataFrame:
        if type_name == "True":
            self.select_true_cash()
        if type_name == "False":
            self.select_false_cash()

    def select_true_cash(self) -> DataFrame:
        self.data_select = self.data[self.data["label"] == 1]

    def select_false_cash(self) -> DataFrame:
        self.data_select = self.data[self.data["label"] == 0]
