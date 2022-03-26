from nltk.tokenize import TweetTokenizer as tokenizer
from .word2vec import word2vec_class
from nltk import FreqDist, ngrams
from numpy import array, empty


class ngram_model:
    def __init__(self, N: int, vocab_max: int = 5000, tokenize: tokenizer = None, embeddings_model: word2vec_class = None) -> None:
        self.tokenize = tokenize if tokenize else self.default_tokenize
        self.punct = set(['.', ',', ';', ':', '-', '^', '»', '!',
                         '¡', '¿', '?', '"', '\'', '...', '<url>',
                          '*', '@usuario'])
        self.N = N
        self.vocab_max = vocab_max
        self.unk = '<unk>'
        self.sos = '<s>'
        self.eos = '</s>'
        self.embeddings_model = embeddings_model

    def get_vocabulary_size(self) -> int:
        return len(self.vocabulary)

    def default_tokenize(self, doc: str) -> list:
        return doc.split("  ")

    def remove_word(self, word: str) -> bool:
        word = word.lower()
        is_punct = word in self.punct
        is_digit = word.isnumeric()
        return is_punct or is_digit

    def sortFreqDisct(self, freq_dist) -> list:
        freq_dict = dict(freq_dist)
        return sorted(freq_dict, key=freq_dict.get, reverse=True)

    def get_vocabulary(self, corpus: list) -> set:
        freq_dist = FreqDist([word.lower()
                              for sentence in corpus
                              for word in self.tokenize(sentence)
                              if not self.remove_word(word)])
        sorted_words = self.sortFreqDisct(freq_dist)[:self.vocab_max - 3]
        return set(sorted_words)

    def fit(self, corpus: list) -> None:
        self.vocabulary = self.get_vocabulary(corpus)
        self.vocabulary.add(self.unk)
        self.vocabulary.add(self.sos)
        self.vocabulary.add(self.eos)
        self.word_index = {}
        self.index_word = {}
        if self.embeddings_model is not None:
            self.embedding_matrix = empty([self.get_vocabulary_size(),
                                           self.embeddings_model.vector_size])
        self.make_data(corpus)

    def make_data(self, corpus: str) -> tuple:
        id = 0
        for doc in corpus:
            for word in self.tokenize(doc):
                word = word.lower()
                if word in self.vocabulary and not word in self.word_index:
                    self.word_index[word] = id
                    self.index_word[id] = word
                    if self.embeddings_model is not None:
                        if word in self.embeddings_model.data:
                            self.embedding_matrix[id] = self.embeddings_model.data[word]
                    id += 1
        # Always add special tokens
        self.word_index.update({
            self.unk: id,
            self.sos: id + 1,
            self.eos: id + 2
        })
        self.index_word.update({
            id: self.unk,
            id + 1: self.sos,
            id + 2: self.eos
        })

    def get_ngram_doc(self, doc: str) -> list:
        doc_tokens = self.tokenize(doc)
        doc_tokens = self.replace_unk(doc_tokens)
        doc_tokens = [word.lower() for word in doc_tokens]
        doc_tokens = [self.sos] * (self.N - 1) + doc_tokens + [self.eos]
        return list(ngrams(doc_tokens, self.N))

    def replace_unk(self, doc_tokens: list) -> list:
        for i, token in enumerate(doc_tokens):
            if token.lower() not in self.vocabulary:
                doc_tokens[i] = self.unk
        return doc_tokens

    def transform(self, corpus: list) -> tuple:
        X_ngrams = []
        y = []
        for doc in corpus:
            doc_ngram = self.get_ngram_doc(doc)
            for words_window in doc_ngram:
                words_window_ids = [self.word_index[word]
                                    for word in words_window]
                X_ngrams.append(list(words_window_ids[:-1]))
                y.append(words_window_ids[-1])
        return array(X_ngrams), array(y)
