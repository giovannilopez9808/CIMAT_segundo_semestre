from sklearn.model_selection import train_test_split
from Modules.Tweet_dataset import TweeterDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from pandas import DataFrame, read_csv
from torch import LongTensor
from argparse import Namespace
from nltk import FreqDist
from os.path import join
from numpy import array


class Mex_data_class:
    def __init__(self, params: dict, args: Namespace) -> None:
        self.params = params
        self.args = args
        self.read()
        self.get_vocabulary()
        self.obtain_loaders()

    def read(self) -> None:
        """
        Lectura de los archivos de datos a partir de su ruta y nombre de archivo
        """
        train_data_filename = join(self.params["path data"],
                                   self.params["train data"])
        train_labels_filename = join(self.params["path data"],
                                     self.params["train labels"])
        test_data_filename = join(self.params["path data"],
                                  self.params["test data"])
        test_labels_filename = join(self.params["path data"],
                                    self.params["test labels"])
        train_text = self.read_file(train_data_filename)
        train_labels = self.read_file(train_labels_filename)
        self.split_data(train_text, train_labels)
        self.test_text = self.read_file(test_data_filename)
        self.test_labels = self.read_file(test_labels_filename)

    def split_data(self, train_data: DataFrame, train_labels: DataFrame) -> None:
        self.train_text,  self.validation_text, self.train_labels, self.validation_labels = train_test_split(
            train_data,
            train_labels,
            test_size=self.args.test_size,
            random_state=self.args.seed)

    def read_file(self, filename: str) -> DataFrame:
        data = read_csv(filename,
                        engine="python",
                        sep="\r\n",
                        header=None)
        data = data[0]
        return data

    def get_vocabulary(self) -> None:
        freq_dist = FreqDist([word.lower()
                              for sentence in self.train_text
                              for word in self.args.tokenize(sentence)])
        max_words = min(self.args.max_vocabulary-1,
                        len(freq_dist))
        sorted_words = self.sortFreqDict(freq_dist)[:max_words]
        self.word_index = {word: i+1
                           for i, word in enumerate(sorted_words)}
        # Append <pad> token with 0 index
        sorted_words.append('<pad>')
        self.word_index['<pad>'] = 0
        self.vocabulary = set(sorted_words)

    def sortFreqDict(self, freq_dist: FreqDist) -> list:
        freq_dict = dict(freq_dist)
        sorted_words = sorted(freq_dict, key=freq_dict.get, reverse=True)
        return sorted_words

    def obtain_loaders(self) -> None:
        print("train")
        self.train_loader = obtain_loader(self.train_text,
                                          self.train_labels,
                                          self.vocabulary,
                                          self.word_index,
                                          self.args)
        print("validation")
        self.validation_loader = obtain_loader(self.validation_text,
                                               self.validation_labels,
                                               self.vocabulary,
                                               self.word_index,
                                               self.args)
        print("test")
        self.test_loader = obtain_loader(self.test_text,
                                         self.test_labels,
                                         self.vocabulary,
                                         self.word_index,
                                         self.args)


def obtain_loader(data: array, labels: array, vocabulary: set, word_index: dict, args: Namespace) -> DataLoader:
    dataset = TweeterDataset(data,
                             labels,
                             vocabulary,
                             word_index,
                             args.tokenize,
                             args.max_seq_len)
    loader = DataLoader(dataset,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                        collate_fn=collate_fn,
                        shuffle=True)
    return loader


def collate_fn(batch):
    # Get X
    batch_tokens = [row[0]
                    for row in batch]
    # Get y
    batch_labels = LongTensor([row[1]
                               for row in batch]).to(float)
    # Pad with 0 (to the rigth) shorter sequences than max_seq_len
    padded_batch_tokens = pad_sequence(batch_tokens,
                                       batch_first=True)
    return padded_batch_tokens, batch_labels.unsqueeze(1)
