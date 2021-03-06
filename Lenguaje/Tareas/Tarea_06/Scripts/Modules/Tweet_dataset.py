from torch.utils.data import Dataset
from pandas import DataFrame
from torch import LongTensor
from typing import Callable


class TweeterDataset(Dataset):
    """
    Modelo de los datos para la red neuronal
    """

    def __init__(self, x: DataFrame, y: DataFrame, vocabulary: set, word_index: dict, tokenizer: Callable[[str], list], max_seq_len: int):
        self.x = x
        self.y = y
        self.vocabulary = vocabulary
        self.word_index = word_index
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        """
        tamaño de los datos
        """
        return self.x.shape[0]

    def __getitem__(self, idx):
        """
        obtiene los tokens de una palabra y su clasificacion dado un indice
        """
        # Get sequence of token ids
        sentence = self.x.iloc[idx]
        tokens = [self.word_index[word]
                  for word in self.tokenizer(sentence)
                  if word in self.vocabulary]
        # Truncate sequence up to max_seq_len
        truncate_len = min(len(tokens), self.max_seq_len)
        tokens = tokens[:truncate_len]
        # Get true label
        label = self.y.iloc[idx]
        return LongTensor(tokens), LongTensor([label])
