from timeit import default_timer
from collections import Counter
from tabulate import tabulate
import re


class hangman_model:
    def __init__(self, path: str) -> None:
        self.letters = 'abcdefghijklmnopqrstuvwxyz'
        self.path = path
        self.read()
        self.obtain_words()
        self.n = sum(self.words.values())

    def read(self,):
        file = open("{}/big.txt".format(self.path), "r",
                    encoding="utf-8")
        self.text = file.read()
        file.close()

    def obtain_words(self):
        words = re.findall(r"\w+", self.text.lower())
        self.words = Counter(words)

    def P(self, word: str) -> float:
        """
        "Probability of `word`."
        """
        return self.words[word] / self.n

    def known(self, words: str) -> set:
        """
        "The subset of `words` that appear in the dictionary of WORDS."
        """
        return set(w for w in words if w in self.words)

    def edits1(self, word: str) -> set:
        """
        All edits that are one edit away from `word`
        """
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        replaces = [L + c + R[1:]
                    for L, R in splits
                    if R for c in self.letters]
        inserts = [L + c + R
                   for L, R in splits
                   for c in self.letters]
        return set(replaces + inserts)

    def edits2(self, word: str):
        """
        All edis that are two edits away from `word`
        """
        return (edit2 for edit1 in self.edits1(word)
                for edit2 in self.edits1(edit1))

    def candidates(self, word: str) -> set:
        """
        Generate possible spelling corrections for word
        """
        return (self.known([word]) or
                self.known(self.edits1(word)) or
                self.known(self.edits2(word)) or
                [word])

    def correction(self, word: str) -> str:
        """
        Most probable spelling correction for word
        """
        return max(self.candidates(word),
                   key=self.P)

    def reduce_incognites(self, word: str) -> list:
        """
        Reduccion de tres incognitas a dos
        ------------
        Input:
        word -> string con maximo tres guiones bajos

        Output:
        word | corrections -> string o lista con las posibles respuestas
        """
        # Creo todos los casos de dos guiones remplazando primer guión
        # por letra del abecedario y resulevo
        posibilities = [word.replace("_", letter, 1)
                        for letter in self.letters]
        # Busco corrección
        corrections = [self.correction(word) for word in posibilities]
        # Me quedo con las que no tengan guiones
        corrections = [word_i for word_i in corrections
                       if not "_" in word_i]
        # Si no encuentro palabra, regreso
        if len(corrections) == 0:
            return word
        return corrections

    def choose_word(self, c):
        return max(c, key=self.P)

    def hangman(self, word) -> str:
        # Busco posiciones de guion
        index = [x.start() for x in re.finditer('_', word)]
        # Si solo me faltan dos letras, ya lo puedo hacer
        if len(index) <= 2:
            return self.correction(word)
        if len(index) == 3:
            # Paso caso de tres a un caso de dos guiones
            corrections = self.reduce_incognites(word)
            # Si no encuentro
            # Elijo de la de mayor probabilidad
            return self.choose_word(corrections)
        if len(index) == 4:
            # Formateamos una leta
            for letter in self.letters:
                # Lo llevo al caso de tres que resolví previamente
                word3 = word.replace("_", letter, 1)
                corrections = self.reduce_incognites(word3)
                # Muestra primera coincidencia
                if not "_" in corrections:
                    return self.choose_word(corrections)
        return 'Only four unknown letters allow.'

    def run(self, words: list) -> None:
        results = []
        for word in words:
            start = default_timer()
            answer = self.hangman(word)
            stop = default_timer()
            time = stop-start
            results += [[word, answer, time]]
        print(tabulate(results,
                       headers=["Palabra original",
                                "Posible respuesta",
                                "Tiempo"]))


hangman = hangman_model("Data/")
words = ["pe_p_e",
         "phi__sop_y",
         "si_nif_c_nc_",
         "w_r_d",
         "thi__",
         "a_w__d",
         "r___",
         "r__d",
         "op_i__",
         "r_vol_t__n"]
hangman.run(words)
