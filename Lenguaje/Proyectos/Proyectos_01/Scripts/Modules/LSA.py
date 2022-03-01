from sklearn.decomposition import TruncatedSVD
from pandas import DataFrame
from numpy import array


class LSA:
    def __init__(self, bow: array, word_index: dict, n_components: int) -> None:
        self.n_components = n_components
        self.word_index = word_index
        self.bow = bow
        self.apply_PCA()

    def apply_PCA(self) -> None:
        self.SVD = TruncatedSVD(self.n_components)
        self.result = self.SVD.fit_transform(self.bow)

    def obtain_words(self):
        index = ["Topic {}".format(i+1) for i in range(self.n_components)]
        self.words = DataFrame(self.SVD.components_,
                               index=index,
                               columns=self.word_index.keys())
        self.words = self.words.transpose()
        self.words.index.name = "Words"
        self.words = self.words.apply(abs)
