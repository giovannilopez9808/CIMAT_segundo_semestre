from pandas import DataFrame, read_csv
from os.path import join


class word2vec_class:
    def __init__(self, params: dict) -> None:
        self.params = params
        self.read()

    def read(self) -> None:
        filename = join(self.params["word2vec path"],
                        self.params["word2vec file"])
        data = read_csv(filename,
                        sep=" ",
                        skiprows=1,
                        header=0,
                        index_col=0)
        data = data.T
        self.vector_size = len(data)
        self.data = self.to_dict(data)

    def to_dict(self, data: DataFrame) -> dict:
        return data.to_dict("list")
