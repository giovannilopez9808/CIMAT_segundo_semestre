from sklearn.decomposition import TruncatedSVD
from pandas import DataFrame
from numpy import array


class LSA:
    def __init__(self, bow: array, words: list, n_components: int) -> None:
        self.n_components = n_components
        self.words = words
        self.bow = bow
        self.apply_PCA()

    def apply_PCA(self) -> None:
        self.SVD = TruncatedSVD(self.n_components)
        self.result = self.SVD.fit_transform(self.bow)

    def obtain_words(self) -> None:
        index = ["Topic {}".format(i+1) for i in range(self.n_components)]
        self.words = DataFrame(self.SVD.components_,
                               index=index,
                               columns=self.words)
        self.words = self.words.transpose()
        self.words.index.name = "Words"
        self.words = self.words.apply(abs)

    def obtain_top_words(self, n_words: int) -> None:
        columns = self.words.columns
        self.top_words = DataFrame()
        for column in columns:
            words_topic = self.words.sort_values(by=column)
            self.top_words[column] = words_topic.index[:n_words]
